#ifndef MNIST_CPP_DATA_H
#define MNIST_CPP_DATA_H

#include <vector>  // 记录数据
#include "stdio.h"
#include "stdint.h"  // 各种 int 类型

class data
{
    std::vector<uint8_t> * feature_vector;  // 数据
    uint8_t label;  // 标签
    int enum_label;  // 标签对应的 int

public:
    data();
    ~data();

    void set_feature_vector(std::vector<uint8_t> *);
    void append_to_feature_vector(uint8_t);
    void set_label(uint8_t);
    void set_enumerated_label(int);

    int get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enumerated_label();
    std::vector<uint8_t> * get_feature_vector();
};



#endif
