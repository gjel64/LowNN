#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include <iostream>
#include <memory>

TEST(TensorTest, Creation) 
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<float>{3.14f, 2.2f});
    EXPECT_EQ(t1->data().get()[0], 3.14f);
    EXPECT_EQ(t1->shape(), std::vector<std::size_t>{2});
    EXPECT_EQ(t1->strides().at(0), 1);
    EXPECT_EQ(t1->strides().size(), 1);
    EXPECT_EQ(t1->size(), 2);

    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(3.14f);
    EXPECT_EQ(t2->data().get()[0], 3.14f);
    EXPECT_EQ(t2->shape(), std::vector<std::size_t>{1});
    EXPECT_EQ(t2->strides(), std::vector<std::size_t>{});
    EXPECT_EQ(t2->size(), 1);

    EXPECT_THROW( 
        std::shared_ptr<Tensor> t3 = std::make_shared<Tensor>(3), 
        std::invalid_argument
    );

    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    std::vector<float> d = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::size_t> s = std::vector<std::size_t>{2, 2};
    std::vector<std::size_t> st = std::vector<std::size_t>{2, 1};
    EXPECT_EQ(t4->data().get()[0], d[0]);
    EXPECT_EQ(t4->shape(), s);
    EXPECT_EQ(t4->strides(), st);
    EXPECT_EQ(t4->size(), 4);

    std::shared_ptr<Tensor> t5 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}});
    std::vector<float> d5 = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<std::size_t> s5 = std::vector<std::size_t>{2, 2, 2};
    std::vector<std::size_t> st5 = std::vector<std::size_t>{4, 2, 1};
    EXPECT_EQ(t5->data().get()[0], d5[0]);
    EXPECT_EQ(t5->shape(), s5);
    EXPECT_EQ(t5->strides(), st5);
    EXPECT_EQ(t5->size(), 8);

    EXPECT_THROW( 
        std::shared_ptr<Tensor> t6 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {2.0f}}), 
        std::invalid_argument
    );

    std::shared_ptr<float[]> data_ptr = std::shared_ptr<float[]>(new float[4]{1.0f, 2.0f, 3.0f, 4.0f});
    std::shared_ptr<Tensor> t7 = std::make_shared<Tensor>(
        data_ptr, std::vector<std::size_t>{2, 2}, 0, std::vector<std::size_t>{2, 1}
    );
    EXPECT_EQ(t7->data().get()[0], 1.0f);
    EXPECT_EQ(t7->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_EQ(t7->strides(), (std::vector<std::size_t>{2, 1}));
    EXPECT_EQ(t7->size(), 4);

}

TEST(TensorTest, BasicMethods)
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<float>{3.14f});
    EXPECT_FLOAT_EQ(t1->item(), 3.14f);

    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f});
    EXPECT_THROW(t2->item(), std::runtime_error);

} 

TEST(TensorTest, Indexing)
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    EXPECT_FLOAT_EQ( ((*t1)[0, 0])->item(), 1.0f );
    EXPECT_FLOAT_EQ( ((*t1)[1, 0])->item(), 3.0f );
    EXPECT_EQ( ((*t1)[1, 0])->data(), t1->data() );
    EXPECT_EQ( ((*t1)[1, 0])->data().get(), t1->data().get() );
    EXPECT_THROW( ((*t1)[2, 0])->item(), std::out_of_range );
    EXPECT_THROW( ((*t1)[1, 0, 0]), std::invalid_argument );

    EXPECT_EQ( ((*t1)[1])->data(), t1->data() );
    EXPECT_EQ( ((*t1)[1])->shape(), std::vector<std::size_t>{2} );
    EXPECT_EQ( ((*t1)[1])->offset(), 2 );
    EXPECT_EQ( ((*t1)[1])->strides(), std::vector<std::size_t>{1} );
    EXPECT_EQ( (*(*t1)[1])[0]->item(), 3.0f );
    EXPECT_EQ( (*(*t1)[1])[1]->item(), 4.0f );

    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}
    });
    EXPECT_FLOAT_EQ( ((*t2)[0, 0, 0])->item(), 1.0f );
    EXPECT_FLOAT_EQ( ((*t2)[1, 0, 0])->item(), 5.0f );
    EXPECT_EQ( ((*t2)[1, 0, 0])->data(), t2->data() );
    EXPECT_EQ( ((*t2)[1, 0, 0])->data().get(), t2->data().get() );
    EXPECT_THROW( ((*t2)[2, 0, 0])->item(), std::out_of_range );
    EXPECT_THROW( ((*t2)[1, 1, 1, 1]), std::invalid_argument );
    EXPECT_EQ( ((*t2)[1])->data(), t2->data() );
    EXPECT_EQ( ((*t2)[1])->shape(), (std::vector<std::size_t>{2, 2}) );
    EXPECT_EQ( ((*t2)[1])->offset(), 4 );
    EXPECT_EQ( ((*t2)[1])->strides(), (std::vector<std::size_t>{2, 1}) );
    EXPECT_EQ( ((*(*t2)[1])[0, 0]->item()), 5.0f );
    EXPECT_EQ( ((*(*t2)[1])[0, 1]->item()), 6.0f );
    EXPECT_EQ( ((*(*t2)[1])[1, 1]->item()), 8.0f );
    EXPECT_EQ( ((*(*t2)[0])[1, 1]->item()), 4.0f );
    EXPECT_EQ( (*(*(*t2)[0])[1])[1]->item(), 4.0f );

}

TEST (TensorTest, Addition)
{

    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{5.0f, 6.0f}, {7.0f, 8.0f}});
    std::shared_ptr<Tensor> t3 = (*t1) + t2;
    EXPECT_EQ(t3->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t3)[0, 0]->item()), 6.0f );
    EXPECT_FLOAT_EQ( ((*t3)[1, 1]->item()), 12.0f );
    
    
    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<float>{7.0f, 8.0f});
    std::shared_ptr<Tensor> t5 = (*t1) + t4;
    EXPECT_EQ(t5->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t5)[0, 0]->item()), 8.0f );
    EXPECT_FLOAT_EQ( ((*t5)[1, 1]->item()), 12.0f );
    
    std::shared_ptr<Tensor> t6 = std::make_shared<Tensor>(10.0f);
    std::shared_ptr<Tensor> t7 = (*t1) + t6;
    EXPECT_EQ(t7->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t7)[0, 0]->item()), 11.0f );
    EXPECT_FLOAT_EQ( ((*t7)[1, 1]->item()), 14.0f );
    
    std::shared_ptr<Tensor> t8 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f}, {2.0f}});
    std::shared_ptr<Tensor> t9 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{3.0f, 4.0f}});
    std::shared_ptr<Tensor> t10 = (*t8) + t9;
    EXPECT_EQ(t10->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t10)[0, 0]->item()), 4.0f );
    EXPECT_FLOAT_EQ( ((*t10)[0, 1]->item()), 5.0f );
    EXPECT_FLOAT_EQ( ((*t10)[1, 0]->item()), 5.0f ); 
    EXPECT_FLOAT_EQ( ((*t10)[1, 1]->item()), 6.0f ); 

    std::shared_ptr<Tensor> t11 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}
    });
    std::shared_ptr<Tensor> t12 = std::make_shared<Tensor>(std::vector<float>{10.0f, 20.0f});
    std::shared_ptr<Tensor> t13 = (*t11) + t12;
    EXPECT_EQ(t13->shape(), (std::vector<std::size_t>{2, 2, 2}));
    EXPECT_FLOAT_EQ( ((*t13)[0, 0, 0]->item()), 11.0f );
    EXPECT_FLOAT_EQ( ((*t13)[0, 0, 1]->item()), 22.0f );
    EXPECT_FLOAT_EQ( ((*t13)[1, 1, 1]->item()), 28.0f );

}

TEST(TensorTest, Multiplication)
{

    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{5.0f, 6.0f}, {7.0f, 8.0f}});
    std::shared_ptr<Tensor> t3 = (*t1) * t2;
    EXPECT_EQ(t3->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t3)[0, 0]->item()), 5.0f );
    EXPECT_FLOAT_EQ( ((*t3)[1, 1]->item()), 32.0f );
    
    
    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<float>{7.0f, 8.0f});
    std::shared_ptr<Tensor> t5 = (*t1) * t4;
    EXPECT_EQ(t5->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t5)[0, 0]->item()), 7.0f );
    EXPECT_FLOAT_EQ( ((*t5)[0, 1]->item()), 16.0f );
    EXPECT_FLOAT_EQ( ((*t5)[1, 0]->item()), 21.0f );
    EXPECT_FLOAT_EQ( ((*t5)[1, 1]->item()), 32.0f );
    
    std::shared_ptr<Tensor> t6 = std::make_shared<Tensor>(10.0f);
    std::shared_ptr<Tensor> t7 = (*t1) * t6;
    EXPECT_EQ(t7->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t7)[0, 0]->item()), 10.0f );
    EXPECT_FLOAT_EQ( ((*t7)[1, 1]->item()), 40.0f );
    
    std::shared_ptr<Tensor> t8 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f}, {2.0f}});
    std::shared_ptr<Tensor> t9 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{3.0f, 4.0f}});
    std::shared_ptr<Tensor> t10 = (*t8) * t9;
    EXPECT_EQ(t10->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t10)[0, 0]->item()), 3.0f );
    EXPECT_FLOAT_EQ( ((*t10)[0, 1]->item()), 4.0f );
    EXPECT_FLOAT_EQ( ((*t10)[1, 0]->item()), 6.0f ); 
    EXPECT_FLOAT_EQ( ((*t10)[1, 1]->item()), 8.0f );

}

TEST(TensorTest, Subtraction)
{

    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{5.0f, 6.0f}, {7.0f, 8.0f}});
    std::shared_ptr<Tensor> t3 = (*t1) - t2;
    EXPECT_EQ(t3->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t3)[0, 0]->item()), -4.0f );
    EXPECT_FLOAT_EQ( ((*t3)[1, 1]->item()), -4.0f );
    
    
    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<float>{7.0f, 8.0f});
    std::shared_ptr<Tensor> t5 = (*t1) - t4;
    EXPECT_EQ(t5->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t5)[0, 0]->item()), -6.0f );
    EXPECT_FLOAT_EQ( ((*t5)[1, 1]->item()), -4.0f );
    
    std::shared_ptr<Tensor> t6 = std::make_shared<Tensor>(10.0f);
    std::shared_ptr<Tensor> t7 = (*t1) - t6;
    EXPECT_EQ(t7->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t7)[0, 0]->item()), -9.0f );
    EXPECT_FLOAT_EQ( ((*t7)[1, 1]->item()), -6.0f );
    
    std::shared_ptr<Tensor> t8 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f}, {2.0f}});
    std::shared_ptr<Tensor> t9 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{3.0f, 4.0f}});
    std::shared_ptr<Tensor> t10 = (*t8) - t9;
    EXPECT_EQ(t10->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t10)[0, 0]->item()), -2.0f );
    EXPECT_FLOAT_EQ( ((*t10)[0, 1]->item()), -3.0f );
    EXPECT_FLOAT_EQ( ((*t10)[1, 0]->item()), -1.0f ); 
    EXPECT_FLOAT_EQ( ((*t10)[1, 1]->item()), -2.0f );

}

TEST(TensorTest, Pow)
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{5.0f, 6.0f}, {7.0f, 8.0f}});
    std::shared_ptr<Tensor> t3 = (*t1).pow(t2);
    EXPECT_EQ(t3->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t3)[0, 0]->item()), 1.0f );
    EXPECT_FLOAT_EQ( ((*t3)[1, 1]->item()), 65536.0f );
    
    
    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<float>{7.0f, 8.0f});
    std::shared_ptr<Tensor> t5 = (*t1).pow(t4);
    EXPECT_EQ(t5->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t5)[0, 0]->item()), 1.0f );
    EXPECT_FLOAT_EQ( ((*t5)[0, 1]->item()), 256.0f );
    EXPECT_FLOAT_EQ( ((*t5)[1, 0]->item()), 2187.0f );
    EXPECT_FLOAT_EQ( ((*t5)[1, 1]->item()), 65536.0f );

    std::shared_ptr<Tensor> t6 = std::make_shared<Tensor>(10.0f);
    std::shared_ptr<Tensor> t7 = (*t1).pow(t6);
    EXPECT_EQ(t7->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t7)[0, 0]->item()), 1.0f );
    EXPECT_FLOAT_EQ( ((*t7)[1, 1]->item()), 1048576.0f );
}

TEST(TensorTest, Division)
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{5.0f, 6.0f}, {7.0f, 8.0f}});
    std::shared_ptr<Tensor> t3 = (*t1) / t2;
    EXPECT_EQ(t3->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t3)[0, 0]->item()), 0.2f );
    EXPECT_FLOAT_EQ( ((*t3)[1, 1]->item()), 0.5f );
    
    
    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<float>{7.0f, 8.0f});
    std::shared_ptr<Tensor> t5 = (*t1) / t4;
    EXPECT_EQ(t5->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t5)[0, 0]->item()), 1.0f/7.0f );
    EXPECT_FLOAT_EQ( ((*t5)[0, 1]->item()), 2.0f/8.0f );
    EXPECT_FLOAT_EQ( ((*t5)[1, 0]->item()), 3.0f/7.0f );
    EXPECT_FLOAT_EQ( ((*t5)[1, 1]->item()), 4.0f/8.0f );
    
    std::shared_ptr<Tensor> t6 = std::make_shared<Tensor>(10.0f);
    std::shared_ptr<Tensor> t7 = (*t1) / t6;
    EXPECT_EQ(t7->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t7)[0, 0]->item()), 0.1f );
    EXPECT_FLOAT_EQ( ((*t7)[1, 1]->item()), 0.4f );
}

TEST(TensorTest, Sqrt)
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 4.0f}, {9.0f, 16.0f}});
    std::shared_ptr<Tensor> t2 = t1->sqrt();
    EXPECT_EQ(t2->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t2)[0, 0]->item()), 1.0f );
    EXPECT_FLOAT_EQ( ((*t2)[1, 1]->item()), 4.0f );
}

TEST(TensorTest, Empty)
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(Tensor::empty(2, 3));
    EXPECT_EQ(t1->shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(t1->size(), 6);

    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(Tensor::empty(2, 3, 3, 4));
    EXPECT_EQ(t2->shape(), (std::vector<std::size_t>{2, 3, 3, 4}));
    EXPECT_EQ(t2->size(), 2*3*3*4);

    std::shared_ptr<Tensor> t3 = std::make_shared<Tensor>(Tensor::empty(1));
    EXPECT_EQ(t3->shape(), (std::vector<std::size_t>{1}));
    EXPECT_EQ(t3->size(), 1);

    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(Tensor::empty());
    EXPECT_EQ(t4->shape(), (std::vector<std::size_t>{}));
    EXPECT_EQ(t4->size(), 0);
}

TEST(TensorTest, Matmul)
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<float>{3.14f, 2.2f});
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}
    }); // 2, 2, 2
    EXPECT_EQ(t1->matmul(t2), nullptr);
    EXPECT_EQ(t2->matmul(t1), nullptr);

    std::shared_ptr<Tensor> t3 = std::make_shared<Tensor>(std::vector<float>{3.14f, 2.2f, 1.1f});
    EXPECT_THROW(t1->matmul(t3),std::invalid_argument);

    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}},{{5.0f, 6.0f}, {7.0f, 8.0f}} 
    }); // 3, 2, 2
    EXPECT_THROW(t4->matmul(t2),std::invalid_argument);

}

TEST(TensorTest, Squeeze)
{   
    std::vector<std::size_t> s1 = {2, 3, 4};
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(Tensor::empty(2, 3, 4));
    std::shared_ptr<Tensor> t2 = t1->squeeze();
    EXPECT_EQ(t2->shape(), s1);
    EXPECT_EQ(t2->size(), 2*3*4);

    std::vector<std::size_t> s2 = {2, 1, 3, 1, 4};
    std::shared_ptr<Tensor> t3 = std::make_shared<Tensor>(Tensor::empty(2, 1, 3, 1, 4));
    std::shared_ptr<Tensor> t4 = t3->squeeze();
    EXPECT_EQ(t4->shape(), s1);
    EXPECT_EQ(t4->size(), 2*1*3*1*4);

}

