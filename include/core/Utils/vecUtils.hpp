#pragma once
#include <vector>
#include <memory>
#include <typeinfo>


template<typename T>
struct is_std_vector : std::false_type {}; // checked at compile
template<typename U, typename Alloc>
struct is_std_vector<std::vector<U, Alloc>> : std::true_type {}; // checked at compile

template<typename T>
struct is_c_array : std::false_type {};
template<typename T, size_t N>
struct is_c_array<T[N]> : std::true_type {}; // taille connue
template<typename T>
struct is_c_array<T[]> : std::true_type {};  // taille inconnue

struct array {
    std::shared_ptr<float[]> data;
    std::size_t size;
};

class VecUtils {
public:
    template<typename V>
    static void flatten_into(const V& v, std::vector<float>& out) {
        if constexpr (is_std_vector<typename V::value_type>::value) {
            for (auto const& sub : v) {
                flatten_into(sub, out);
            }
        } else {
            out.reserve(out.size() + v.size());
            for (auto const& x : v) {
                out.push_back(static_cast<float>(x));
            }
        }
    }

    template<typename V>
    static std::vector<size_t> get_shape(const V& v) {
        std::vector<size_t> shape = { v.size() };

        if constexpr (is_std_vector<typename V::value_type>::value) {
            if (!v.empty()){
                auto sub_shape = get_shape(v[0]);
                shape.insert(shape.end(), sub_shape.begin(), sub_shape.end());
            }
        }
        
        return shape;
    } 

    static std::vector<size_t> broadcast(const std::vector<size_t>& shape1, std::vector<size_t> shape2);

    static std::vector<size_t> squeeze_shape(std::vector<size_t> shape);
};