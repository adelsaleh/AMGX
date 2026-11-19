// SPDX-License-Identifier: BSD-3-Clause
#include "unit_test.h"
#include "truncate_max_elements_oracle.h"

namespace amgx
{
DECLARE_UNITTEST_BEGIN(TruncateMaxElements);
void run()
{
    TruncateMaxElementsOracle<TConfig> oracle;
    const char *fixtures[] = {"unique", "signed", "ties", "signed_ties", "cancel", "mixed"};
    for (int cap = 1; cap <= 32; ++cap)
    {
        for (const char *fixture : fixtures)
        {
            if (std::string(fixture) == "cancel" && cap % 2) { continue; }
            oracle.check_fixture(257, cap, fixture);
        }
    }
    // Small matrices and partial 32/64/128-thread blocks, including empty rows.
    const int counts[] = {1, 31, 32, 33, 63, 64, 65, 127, 128, 129};
    for (int cap : {8, 9, 16, 17, 32})
    {
        for (int rows : counts)
        {
            oracle.check_fixture(rows, cap, "unique");
            oracle.check_fixture(rows, cap, "mixed");
        }
    }
    Matrix<TConfig> matrix(1, 1, 0, CSR);
    bool rejected = false;
    try { Truncate<TConfig>::truncateByMaxElements(matrix, 33); }
    catch (amgx_exception &error) { rejected = error.reason() == AMGX_ERR_BAD_PARAMETERS; }
    UNITTEST_ASSERT_TRUE(rejected);
}
DECLARE_UNITTEST_END(TruncateMaxElements);
TruncateMaxElements<TemplateMode<AMGX_mode_dDDI>::Type> truncate_max_dDDI;
TruncateMaxElements<TemplateMode<AMGX_mode_dDFI>::Type> truncate_max_dDFI;
TruncateMaxElements<TemplateMode<AMGX_mode_dFFI>::Type> truncate_max_dFFI;
} // namespace amgx
