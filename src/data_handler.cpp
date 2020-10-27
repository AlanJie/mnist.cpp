#include "../include/data_handler.h"

data_handler::data_handler()
{
    data_array = new std::vector<data *>;
    training_data = new std::vector<data *>;
    validation_data = new std::vector<data *>;
    test_data = new std::vector<data *>;
}

data_handler::~data_handler()
{
    // Free Dynamically Allocated STUFF
}

void data_handler::read_feature_vector(std::string path)
{
    uint32_t header[4];  // |MAGIC|NUM_IMAGES|ROWSIZE|COLSIZE  存储训练集信息
    unsigned char bytes[4];
    // c_str()函数返回一个指向正规C字符串的指针常量, 内容与本string串相同.
    //这是为了与c语言兼容，在c语言中没有string类型，故必须通过string类对象的成员函数c_str()把string 对象转换成c中的字符串样式。
    FILE *f = fopen(path.c_str(), "r");
    if(f)  // 如果成功读到文件内容
    {
        for(int i=0; i<4; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f))
            {
                // header 存储了训练集的信息
                // 训练集中4个元素分别为 magic number, number of images(60000), number of rows(28), number of columns(28)
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting file header.\n");

        int image_size = header[2] * header[3];  // 28 * 28

        for(int i=0; i<header[1]; i++)  // 循环读取 60000 张图片
        {
            data *d = new data();  // 读取一张图片
            uint8_t elements[1];  // 存储当前位置的像素值
            for(int j=0; j<image_size; j++)
            {
                if(fread(elements, sizeof(elements), 1, f))
                {
                    d->append_to_feature_vector(elements[0]);
                } else
                {
                    printf("Error Reading from FILE\n");
                    exit(1);
                }
            }
            data_array->push_back(d);
        }
        printf("Successfully read and stored %lu feature vectors.\n", data_array->size());
    } else
    {
        printf("Could not find file.\n");
        exit(1);
    }

}

void data_handler::read_feature_labels(std::string path)
{
    uint32_t header[2];  // |MAGIC|NUM ITEMS|
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");

    if(f)
    {
        for(int i=0; i<2; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting Label file Header.\n");

        for(int i=0; i<header[1]; i++)  // 60000
        {
            uint8_t element[1];  // 存储当前的label
            if(fread(element, sizeof(element), 1, f))
            {
                data_array->at(i)->set_label(element[0]);
            } else
            {
                printf("ERROR Reading from file.\n");
                exit(1);
            }
        }
        printf("Successfully read and stored label.\n");
    } else
    {
        printf("Cound not find file.\n");
        exit(1);
    }
}

void data_handler::split_data()
{
    std::unordered_set<int> used_indexes;  // 记录已经添加过的数据
    int train_size = data_array->size() * TRAIN_SET_PERCENT;
    int valid_size = data_array->size() * VALIDATION_PERCENT;
    int test_size = data_array->size() * TEST_SET_PERCENT;

    // Training data
    int count = 0;
    while(count < train_size)
    {
        int rand_index = rand() % data_array->size();  // [0, data_array->size()-1]
        if(used_indexes.find(rand_index) == used_indexes.end())  // 没有找到，则返回最后一个元素的指针?
        {
            training_data->push_back(data_array->at(rand_index)); // 如果没有添加过，则将该元素添加到对应的数据列表
            used_indexes.insert(rand_index);
            count++;
        }
    }

    // Test data
    count = 0;
    while(count < test_size)
    {
        int rand_index = rand() % data_array->size();
        if(used_indexes.find(rand_index) == used_indexes.end())
        {
            test_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }

    // Validation data
    count = 0;
    while(count < valid_size)
    {
        int rand_index = rand() % data_array->size();
        if(used_indexes.find(rand_index) == used_indexes.end())
        {
            validation_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }

    printf("Training Data Size: %lu.\n", training_data->size());
    printf("Test Data Size: %lu.\n", test_data->size());
    printf("Validation Data Size: %lu.\n", validation_data->size());
}

void data_handler::count_classes()
{
    int count = 0;
    for(unsigned i=0; i<data_array->size(); i++)  // 60000
    {
        if(class_map.find(data_array->at(i)->get_label()) == class_map.end())
        {
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enumerated_label(count);
            count++;
        }
    }
    num_classes = count;
    printf("Successfully Extracted %d Unique Classes.\n", num_classes);
}

uint32_t data_handler::convert_to_little_endian(const unsigned char* bytes)
{
    return (uint32_t) ((bytes[0] << 24 | bytes[1] << 16 | bytes[2] << 8 | bytes[3]));
}

std::vector<data *> * data_handler::get_training_data()
{
    return training_data;
}

std::vector<data *> * data_handler::get_test_data()
{
    return test_data;
}

std::vector<data *> * data_handler::get_validation_data()
{
    return validation_data;
}


int main()
{
    data_handler *dh = new data_handler();
    dh->read_feature_vector("./mnist/train-images-idx3-ubyte");
    dh->read_feature_labels("./mnist/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();
}