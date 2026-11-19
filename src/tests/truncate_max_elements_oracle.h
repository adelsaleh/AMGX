// SPDX-License-Identifier: BSD-3-Clause
// Shared independent CPU oracle for the native regression and isolated checker.
#pragma once
#include <matrix.h>
#include <truncate.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace amgx
{
template<class TConfig>
class TruncateMaxElementsOracle
{
public:
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    typedef TemplateConfig<AMGX_device, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_d;
    typedef typename TConfig::MatPrec ValueTypeA;
    typedef typename TConfig::IndPrec IndexType;
    typedef Matrix<TConfig_h> Matrix_h;
    typedef Matrix<TConfig_d> Matrix_d;
    std::string failure_context;
    void PrintOnFail(const std::string &text) { failure_context = text; }
    void check(bool condition, const char *expression, const char *file, int line)
    {
        if (!condition)
        {
            std::ostringstream message;
            message << failure_context << " failed: " << expression << " at " << file << ':' << line;
            throw std::runtime_error(message.str());
        }
    }
#define TRUNCATION_CHECK(expression) check(bool(expression), #expression, __FILE__, __LINE__)
#define TRUNCATION_EQUAL(left, right) TRUNCATION_CHECK((left) == (right))
int integer_option(const char *name, int fallback)
{
    const char *text = std::getenv(name);
    if (text == NULL) { return fallback; }
    char *end = NULL;
    const long value = std::strtol(text, &end, 10);
    TRUNCATION_CHECK(end != text && *end == '\0');
    TRUNCATION_CHECK(value > 0 && value <= 1000000);
    return static_cast<int>(value);
}

typedef std::pair<int, ValueTypeA> Entry;

void check_fixture(int nrows, int cap, const std::string &kind)
{
    const int full_length = 65; // Every ordinary row is longer than every supported cap.
    std::vector<int> lengths(nrows, full_length);
    if (kind == "mixed")
    {
        const int choices[] = {0, 1, cap, cap + 1, full_length};
        for (int row = 0; row < nrows; ++row) { lengths[row] = choices[row % 5]; }
    }
    int nnz = 0;
    for (int row = 0; row < nrows; ++row) { nnz += lengths[row]; }
    const int ncols = std::max(nrows, 257); // Also exercise partial blocks on rectangular P.
    Matrix_h input(nrows, ncols, nnz, CSR);
    input.set_initialized(0);
    int offset = 0;
    input.row_offsets[0] = 0;
    for (int row = 0; row < nrows; ++row)
    {
        std::vector<Entry> entries;
        for (int j = 0; j < lengths[row]; ++j)
        {
            ValueTypeA value = ValueTypeA(j + 1) + ValueTypeA(row % 5) / ValueTypeA(128);
            if (kind == "signed" && j % 2) { value = -value; }
            if (kind == "ties" || kind == "signed_ties")
            {
                value = ValueTypeA(10 + j % 5);
                if (kind == "signed_ties" && j % 2) { value = -value; }
            }
            if (kind == "cancel")
            {
                // For even caps, the retained sum is exactly zero although the
                // original row sum is positive. This checks multiplier=1 semantics.
                value = j < cap ? ValueTypeA((j % 2 ? -1 : 1) * (1000 + j / 2))
                                : ValueTypeA(0.25) + ValueTypeA(j - cap) / ValueTypeA(128);
            }
            entries.push_back(Entry((row + 3 * j) % ncols, value));
        }
        std::sort(entries.begin(), entries.end(),
                  [](const Entry &a, const Entry &b) { return a.first < b.first; });
        for (size_t j = 0; j < entries.size(); ++j)
        {
            input.col_indices[offset] = entries[j].first;
            input.values[offset] = entries[j].second;
            ++offset;
        }
        input.row_offsets[row + 1] = offset;
    }
    input.computeDiagonal();
    input.set_initialized(1);

    Matrix_d device_matrix = input;
    Truncate<TConfig>::truncateByMaxElements(device_matrix, cap);
    TRUNCATION_EQUAL(cudaDeviceSynchronize(), cudaSuccess);
    Matrix_h output = device_matrix;
    TRUNCATION_EQUAL(output.get_num_rows(), nrows);
    TRUNCATION_EQUAL(output.get_num_cols(), ncols);

    // Independent oracle: stable full-row sort, not the kernel's bounded
    // insertion algorithm. Accumulation and scaling use MATRIX precision.
    const double eps = double(std::numeric_limits<ValueTypeA>::epsilon());
    int expected_nnz = 0;
    std::uint64_t output_hash = UINT64_C(14695981039346656037);
    const auto hash_bytes = [&output_hash](const void *pointer, size_t size)
    {
        const unsigned char *bytes = static_cast<const unsigned char *>(pointer);
        for (size_t j = 0; j < size; ++j)
        {
            output_hash ^= bytes[j];
            output_hash *= UINT64_C(1099511628211);
        }
    };
    for (int row = 0; row < nrows; ++row)
    {
        std::vector<Entry> selected;
        ValueTypeA old_sum = ValueTypeA(0);
        for (int j = input.row_offsets[row]; j < input.row_offsets[row + 1]; ++j)
        {
            selected.push_back(Entry(input.col_indices[j], input.values[j]));
            old_sum += input.values[j];
        }
        if (int(selected.size()) > cap)
        {
            std::stable_sort(selected.begin(), selected.end(),
                [](const Entry &a, const Entry &b)
                { return std::abs(a.second) > std::abs(b.second); });
            selected.resize(cap);
        }
        ValueTypeA retained_sum = ValueTypeA(0);
        for (size_t j = 0; j < selected.size(); ++j) { retained_sum += selected[j].second; }
        const ValueTypeA multiplier = retained_sum == ValueTypeA(0)
                                   ? ValueTypeA(1) : old_sum / retained_sum;
        std::ostringstream label;
        label << "fixture=" << kind << " cap=" << cap << " row=" << row;
        this->PrintOnFail(label.str());
        TRUNCATION_EQUAL(output.row_offsets[row], expected_nnz);
        expected_nnz += int(selected.size());
        TRUNCATION_EQUAL(output.row_offsets[row + 1], expected_nnz);
        double scaled_sum = 0.0, scaled_abs_sum = 0.0;
        for (size_t j = 0; j < selected.size(); ++j)
        {
            const int output_index = output.row_offsets[row] + int(j);
            const ValueTypeA expected = selected[j].second * multiplier;
            const ValueTypeA actual = output.values[output_index];
            TRUNCATION_EQUAL(output.col_indices[output_index], selected[j].first);
            TRUNCATION_CHECK(std::isfinite(double(actual)));
            TRUNCATION_CHECK(std::abs(double(actual) - double(expected)) <=
                                 32 * eps * std::max(1.0, std::abs(double(expected))));
            scaled_sum += double(actual);
            scaled_abs_sum += std::abs(double(actual));
        }
        if (retained_sum != ValueTypeA(0))
        {
            TRUNCATION_CHECK(std::abs(scaled_sum - double(old_sum)) <=
                                 128 * eps * std::max(1.0, scaled_abs_sum));
        }
    }
    TRUNCATION_EQUAL(output.get_num_nz(), expected_nnz);
    hash_bytes(output.row_offsets.raw(), (nrows + 1) * sizeof(IndexType));
    hash_bytes(output.col_indices.raw(), expected_nnz * sizeof(IndexType));
    hash_bytes(output.values.raw(), expected_nnz * sizeof(ValueTypeA));
    std::cout << "TruncateMaxElementsOracle fixture=" << kind << " rows=" << nrows
              << " cap=" << cap << " matrix_bytes=" << sizeof(ValueTypeA)
              << " output_fnv1a64=" << output_hash << std::endl;
}

void run()
{
    const int cap = integer_option("AMGX_TRUNCATION_TEST_CAP", 32);
    const int rows = integer_option("AMGX_TRUNCATION_TEST_ROWS", 257);
    TRUNCATION_CHECK(cap >= 1 && cap <= 32);
    TRUNCATION_CHECK(rows >= 1);
    const char *choice = std::getenv("AMGX_TRUNCATION_TEST_CASE");
    const std::string selected = choice == NULL ? "all" : choice;
    const char *fixtures[] = {"unique", "signed", "ties", "signed_ties", "cancel", "mixed"};
    bool ran = false;
    for (int fixture = 0; fixture < 6; ++fixture)
    {
        const std::string kind(fixtures[fixture]);
        if (selected != "all" && selected != kind) { continue; }
        if (kind == "cancel" && cap % 2) { continue; }
        check_fixture(rows, cap, kind);
        ran = true;
    }
    TRUNCATION_CHECK(ran);
}

};
#undef TRUNCATION_EQUAL
#undef TRUNCATION_CHECK

} // namespace amgx
