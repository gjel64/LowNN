#pragma once
#include <vector>


template<typename T>
struct is_std_vector : std::false_type {}; // checked at compile
template<typename U, typename Alloc>
struct is_std_vector<std::vector<U, Alloc>> : std::true_type {}; // checked at compile

class VecUtils {
public:
    template<typename V>
    static void flatten_into(const V& v, std::vector<float>& out) {
        if constexpr (is_std_vector<typename V::value_type>::value) {
            for (auto const& sub : v) flatten_into(sub, out);
        } else {
            out.reserve(out.size() + v.size());
            for (auto const& x : v) out.push_back(static_cast<float>(x));
        }
    }

};