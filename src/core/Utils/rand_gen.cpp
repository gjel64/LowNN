#include "core/Utils/rand_gen.hpp"

RandomGenerator* RandomGenerator::_instance = nullptr;


RandomGenerator* RandomGenerator::getInstance() {
    if (_instance == nullptr) {
        _instance = new RandomGenerator();
    }
    return _instance;
}

float RandomGenerator::rand_dist(float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    return dist(_rng);
}
