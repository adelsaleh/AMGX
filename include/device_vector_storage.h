// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "vector_thrust_allocator.h"
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <type_traits>

namespace amgx
{

// Keep the owning container private: converting a borrowed vector to a Thrust
// vector base would silently expose the (empty) owning allocation instead.
template<class T>
class device_vector_storage
{
    using owned_type = device_vector_alloc<T>;
    owned_type owned_;
    T *borrowed_ = nullptr;
    size_t borrowed_size_ = 0;
    bool attached_ = false;

    void require_owned() const
    {
        if (attached_)
            FatalError("Cannot resize, swap, or replace borrowed vector storage; detach first",
                       AMGX_ERR_BAD_PARAMETERS);
    }

public:
    using value_type = T;
    using size_type = typename owned_type::size_type;
    using difference_type = typename owned_type::difference_type;
    using pointer = typename owned_type::pointer;
    using const_pointer = typename owned_type::const_pointer;
    using reference = typename owned_type::reference;
    using const_reference = typename owned_type::const_reference;
    using iterator = typename owned_type::iterator;
    using const_iterator = typename owned_type::const_iterator;
    using reverse_iterator = typename owned_type::reverse_iterator;
    using const_reverse_iterator = typename owned_type::const_reverse_iterator;

    device_vector_storage() = default;
    explicit device_vector_storage(size_type n) : owned_(n) {}
    device_vector_storage(size_type n, const T &value) : owned_(n, value) {}
    device_vector_storage(const device_vector_storage &other)
        : owned_(other.begin(), other.end()) {}
    template<class Other, typename std::enable_if<!std::is_integral<Other>::value, int>::type = 0>
    explicit device_vector_storage(const Other &other)
        : owned_(other.begin(), other.end()) {}
    device_vector_storage &operator=(const device_vector_storage &other)
    {
        if (this != &other) assign(other.begin(), other.end());
        return *this;
    }

    bool is_borrowed() const { return attached_; }
    void attach_storage(T *ptr, size_type n)
    {
        require_owned();
        // Attachment is deliberately explicit; no implicit discard of uploads.
        if (!owned_.empty())
            FatalError("Attach requires an empty vector", AMGX_ERR_BAD_PARAMETERS);
        owned_.shrink_to_fit();
        borrowed_ = ptr;
        borrowed_size_ = n;
        attached_ = true;
    }
    void detach_storage()
    {
        borrowed_ = nullptr;
        borrowed_size_ = 0;
        attached_ = false;
    }

    size_type size() const { return attached_ ? borrowed_size_ : owned_.size(); }
    size_type capacity() const { return attached_ ? borrowed_size_ : owned_.capacity(); }
    size_type max_size() const { return owned_.max_size(); }
    bool empty() const { return size() == 0; }
    template<class U>
    bool operator==(const device_vector_storage<U> &other) const
    {
        // Compare the active iterators, including borrowed allocations. Comparing
        // owned_ alone would make every borrowed vector look empty and equal.
        return size() == other.size() &&
            (empty() || amgx::thrust::equal(begin(), end(), other.begin()));
    }
    template<class U>
    bool operator!=(const device_vector_storage<U> &other) const
    {
        return !(*this == other);
    }
    pointer data() { return attached_ ? pointer(borrowed_) : owned_.data(); }
    const_pointer data() const { return attached_ ? const_pointer(borrowed_) : owned_.data(); }
    iterator begin() { return iterator(data()); }
    const_iterator begin() const { return const_iterator(data()); }
    iterator end() { return empty() ? begin() : begin() + size(); }
    const_iterator end() const { return empty() ? begin() : begin() + size(); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
    reference operator[](size_type i) { return data()[i]; }
    const_reference operator[](size_type i) const { return data()[i]; }
    reference front() { return (*this)[0]; }
    const_reference front() const { return (*this)[0]; }
    reference back() { return (*this)[size()-1]; }
    const_reference back() const { return (*this)[size()-1]; }

    void resize(size_type n)
    {
        if (attached_) { if (n != size()) require_owned(); }
        else owned_.resize(n);
    }
    void resize(size_type n, const T &value)
    {
        if (attached_) { if (n != size()) require_owned(); }
        else owned_.resize(n, value);
    }
    void reserve(size_type n) { require_owned(); owned_.reserve(n); }
    void shrink_to_fit() { require_owned(); owned_.shrink_to_fit(); }
    void clear() { require_owned(); owned_.clear(); }
    void swap(device_vector_storage &other)
    {
        require_owned(); other.require_owned(); owned_.swap(other.owned_);
    }
    template<class Iterator, typename std::enable_if<!std::is_integral<Iterator>::value, int>::type = 0>
    void assign(Iterator first, Iterator last)
    {
        if (!attached_) { owned_.assign(first, last); return; }
        if (static_cast<size_type>(amgx::thrust::distance(first, last)) != size()) require_owned();
        amgx::thrust::copy(first, last, begin());
    }
    void assign(size_type n, const T &value)
    {
        if (!attached_) { owned_.assign(n, value); return; }
        if (n != size()) require_owned();
        amgx::thrust::fill(begin(), end(), value);
    }
    void push_back(const T &value) { require_owned(); owned_.push_back(value); }
    void pop_back() { require_owned(); owned_.pop_back(); }
    iterator erase(iterator position) { require_owned(); return owned_.erase(position); }
    iterator erase(iterator first, iterator last) { require_owned(); return owned_.erase(first, last); }
    template<class... Args>
    void insert(iterator position, Args... args) { require_owned(); owned_.insert(position, args...); }
};
} // namespace amgx
