#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include <algorithm>
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

    std::shared_ptr<Tensor> t8 = std::make_shared<Tensor>( Tensor::fill(1.0f, {2, 3, 4}) );
    EXPECT_EQ(t8->shape(), (std::vector<std::size_t>{2, 3, 4}));
    EXPECT_EQ(t8->size(), 24);
    for (std::size_t i = 0; i < t8->size(); i++) {
        EXPECT_FLOAT_EQ(t8->data().get()[i], 1.0f);
    }

} 

TEST(TensorTest, Indexing)
{

    std::shared_ptr<Tensor> t = std::make_shared<Tensor>(std::vector<float>{3.14f});
    EXPECT_FLOAT_EQ(t->item(), 3.14f);

    std::shared_ptr<Tensor> tt = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f});
    EXPECT_THROW(tt->item(), std::runtime_error);


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

TEST(TensorTest, Matmul)
{
    
    // 1D x 2D
    std::shared_ptr<Tensor> t13 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f});
    std::shared_ptr<Tensor> t14 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    std::shared_ptr<Tensor> t15 = (*t13).matmul(t14);
    EXPECT_EQ(t15->shape(), (std::vector<std::size_t>{3}));
    EXPECT_FLOAT_EQ( ((*t15)[0]->item()), 9.0f );
    EXPECT_FLOAT_EQ( ((*t15)[1]->item()), 12.0f );
    EXPECT_FLOAT_EQ( ((*t15)[2]->item()), 15.0f );
    
    // 3D x 2D
    std::shared_ptr<Tensor> t22 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 2.0f}, {3.0f, 4.0f}},
        {{5.0f, 6.0f}, {7.0f, 8.0f}}
    });
    std::shared_ptr<Tensor> t23 = std::make_shared<Tensor>(std::vector<std::vector<float>>{
        {1.0f, 0.0f},
        {0.0f, 1.0f}
    });
    std::shared_ptr<Tensor> t24 = (*t22).matmul(t23);
    EXPECT_EQ(t24->shape(), (std::vector<std::size_t>{2, 2, 2}));
    EXPECT_FLOAT_EQ( ((*t24)[0, 0, 0]->item()), 1.0f );
    EXPECT_FLOAT_EQ( ((*t24)[0, 0, 1]->item()), 2.0f );
    EXPECT_FLOAT_EQ( ((*t24)[0, 1, 0]->item()), 3.0f );
    EXPECT_FLOAT_EQ( ((*t24)[0, 1, 1]->item()), 4.0f );
    EXPECT_FLOAT_EQ( ((*t24)[1, 0, 0]->item()), 5.0f );
    EXPECT_FLOAT_EQ( ((*t24)[1, 0, 1]->item()), 6.0f );
    EXPECT_FLOAT_EQ( ((*t24)[1, 1, 0]->item()), 7.0f );
    EXPECT_FLOAT_EQ( ((*t24)[1, 1, 1]->item()), 8.0f );

    // 2D*2D
    std::shared_ptr<Tensor> t10 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    std::shared_ptr<Tensor> t11 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{5.0f, 6.0f}, {7.0f, 8.0f}});
    std::shared_ptr<Tensor> t12 = (*t10).matmul(t11);
    EXPECT_FLOAT_EQ( ((*t12)[0, 0]->item()), 19.0f );
    EXPECT_FLOAT_EQ( ((*t12)[0, 1]->item()), 22.0f );
    EXPECT_FLOAT_EQ( ((*t12)[1, 0]->item()), 43.0f );
    EXPECT_FLOAT_EQ( ((*t12)[1, 1]->item()), 50.0f );

    // 2D x 1D
    std::shared_ptr<Tensor> t16 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    std::shared_ptr<Tensor> t17 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f});
    std::shared_ptr<Tensor> t18 = (*t16).matmul(t17);
    EXPECT_EQ(t18->shape(), (std::vector<std::size_t>{2}));
    EXPECT_FLOAT_EQ( ((*t18)[0]->item()), 14.0f );
    EXPECT_FLOAT_EQ( ((*t18)[1]->item()), 32.0f );

    // 3D x 3D
    std::shared_ptr<Tensor> t19 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 2.0f}, {3.0f, 4.0f}},
        {{5.0f, 6.0f}, {7.0f, 8.0f}}
    });
    std::shared_ptr<Tensor> t20 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 0.0f}, {0.0f, 1.0f}},
        {{0.0f, 1.0f}, {1.0f, 0.0f}}
    });
    std::shared_ptr<Tensor> t21 = (*t19).matmul(t20);
    EXPECT_EQ(t21->shape(), (std::vector<std::size_t>{2, 2, 2}));
    EXPECT_FLOAT_EQ( ((*t21)[0, 0, 0]->item()), 1.0f );
    EXPECT_FLOAT_EQ( ((*t21)[0, 0, 1]->item()), 2.0f );
    EXPECT_FLOAT_EQ( ((*t21)[0, 1, 0]->item()), 3.0f );
    EXPECT_FLOAT_EQ( ((*t21)[0, 1, 1]->item()), 4.0f );
    EXPECT_FLOAT_EQ( ((*t21)[1, 0, 0]->item()), 6.0f );
    EXPECT_FLOAT_EQ( ((*t21)[1, 0, 1]->item()), 5.0f );
    EXPECT_FLOAT_EQ( ((*t21)[1, 1, 0]->item()), 8.0f );
    EXPECT_FLOAT_EQ( ((*t21)[1, 1, 1]->item()), 7.0f );

    // 4D x 3D
    std::shared_ptr<Tensor> t25 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<std::vector<float>>>>{
        {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}},
        {{{9.0f, 10.0f}, {11.0f, 12.0f}}, {{13.0f, 14.0f}, {15.0f, 16.0f}}}
    });
    std::shared_ptr<Tensor> t26 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 0.0f}, {0.0f, 1.0f}},
        {{1.0f, 0.0f}, {0.0f, 1.0f}}
    });
    std::shared_ptr<Tensor> t27 = (*t25).matmul(t26);
    EXPECT_EQ(t27->shape(), (std::vector<std::size_t>{2, 2, 2, 2}));
    EXPECT_FLOAT_EQ( ((*t27)[0, 0, 0, 0]->item()), 1.0f );
    EXPECT_FLOAT_EQ( ((*t27)[0, 0, 0, 1]->item()), 2.0f );
    EXPECT_FLOAT_EQ( ((*t27)[0, 0, 1, 0]->item()), 3.0f );
    EXPECT_FLOAT_EQ( ((*t27)[0, 0, 1, 1]->item()), 4.0f );
    EXPECT_FLOAT_EQ( ((*t27)[0, 1, 0, 0]->item()), 5.0f );
    EXPECT_FLOAT_EQ( ((*t27)[0, 1, 0, 1]->item()), 6.0f );
    EXPECT_FLOAT_EQ( ((*t27)[0, 1, 1, 0]->item()), 7.0f );
    EXPECT_FLOAT_EQ( ((*t27)[0, 1, 1, 1]->item()), 8.0f );
    EXPECT_FLOAT_EQ( ((*t27)[1, 0, 0, 0]->item()), 9.0f );
    EXPECT_FLOAT_EQ( ((*t27)[1, 0, 0, 1]->item()), 10.0f );
    EXPECT_FLOAT_EQ( ((*t27)[1, 0, 1, 0]->item()), 11.0f );
    EXPECT_FLOAT_EQ( ((*t27)[1, 0, 1, 1]->item()), 12.0f );
    EXPECT_FLOAT_EQ( ((*t27)[1, 1, 0, 0]->item()), 13.0f );
    EXPECT_FLOAT_EQ( ((*t27)[1, 1, 0, 1]->item()), 14.0f );
    EXPECT_FLOAT_EQ( ((*t27)[1, 1, 1, 0]->item()), 15.0f );
    EXPECT_FLOAT_EQ( ((*t27)[1, 1, 1, 1]->item()), 16.0f );
} 

TEST(TensorTest, Sum)
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    std::shared_ptr<Tensor> t2 = t1->sum(0);
    EXPECT_EQ(t2->shape(), (std::vector<std::size_t>{2}));
    EXPECT_FLOAT_EQ( ((*t2)[0]->item()), 4.0f );
    EXPECT_FLOAT_EQ( ((*t2)[1]->item()), 6.0f );

    std::shared_ptr<Tensor> t3 = t1->sum(1);
    EXPECT_EQ(t3->shape(), (std::vector<std::size_t>{2}));
    EXPECT_FLOAT_EQ( ((*t3)[0]->item()), 3.0f );
    EXPECT_FLOAT_EQ( ((*t3)[1]->item()), 7.0f );


    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 2.0f}, {3.0f, 4.0f}},
        {{5.0f, 6.0f}, {7.0f, 8.0f}}
    });
    std::shared_ptr<Tensor> t5 = t4->sum(0);
    EXPECT_EQ(t5->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t5)[0, 0]->item()), 6.0f );
    EXPECT_FLOAT_EQ( ((*t5)[0, 1]->item()), 8.0f );
    EXPECT_FLOAT_EQ( ((*t5)[1, 0]->item()), 10.0f );
    EXPECT_FLOAT_EQ( ((*t5)[1, 1]->item()), 12.0f );

    std::shared_ptr<Tensor> t6 = t4->sum(1);
    EXPECT_EQ(t6->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t6)[0, 0]->item()), 4.0f );
    EXPECT_FLOAT_EQ( ((*t6)[0, 1]->item()), 6.0f );
    EXPECT_FLOAT_EQ( ((*t6)[1, 0]->item()), 12.0f );
    EXPECT_FLOAT_EQ( ((*t6)[1, 1]->item()), 14.0f );

    std::shared_ptr<Tensor> t7 = t4->sum(2);
    EXPECT_EQ(t7->shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ( ((*t7)[0, 0]->item()), 3.0f );
    EXPECT_FLOAT_EQ( ((*t7)[0, 1]->item()), 7.0f );
    EXPECT_FLOAT_EQ( ((*t7)[1, 0]->item()), 11.0f );
    EXPECT_FLOAT_EQ( ((*t7)[1, 1]->item()), 15.0f );

    std::shared_ptr<Tensor> t8 = t4->sum();
    EXPECT_EQ(t8->shape(), (std::vector<std::size_t>{1}));
    EXPECT_FLOAT_EQ( t8->item(), 36.0f );
}

TEST(TensorTest, PowAndAbsAndRelu)
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f}, true);
    std::shared_ptr<Tensor> t2 = t1->pow(2.0f);
    EXPECT_EQ(t2->shape(), (std::vector<std::size_t>{2}));
    EXPECT_FLOAT_EQ( ((*t2)[0]->item()), 1.0f );
    EXPECT_FLOAT_EQ( ((*t2)[1]->item()), 4.0f );


    std::shared_ptr<Tensor> t3 = std::make_shared<Tensor>(std::vector<float>{-1.0f, 2.0f}, true);
    std::shared_ptr<Tensor> t4 = t3->abs();
    EXPECT_EQ(t4->shape(), (std::vector<std::size_t>{2}));
    EXPECT_FLOAT_EQ( ((*t4)[0]->item()), 1.0f );
    EXPECT_FLOAT_EQ( ((*t4)[1]->item()), 2.0f );

    std::shared_ptr<Tensor> t5 = std::make_shared<Tensor>(std::vector<float>{-1.0f, 2.0f}, true);
    std::shared_ptr<Tensor> t6 = t5->relu();
    EXPECT_EQ(t6->shape(), (std::vector<std::size_t>{2}));
    EXPECT_FLOAT_EQ( ((*t6)[0]->item()), 0.0f );
    EXPECT_FLOAT_EQ( ((*t6)[1]->item()), 2.0f );

}

TEST(TensorTest, BackpropAdd)
{
    
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(2.0f, true);
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(3.0f, true);
    std::shared_ptr<Tensor> t3 = (*t1) + t2;
    t3->backward();

    EXPECT_FLOAT_EQ( t1->gradp().get()[0], 1.0f );
    EXPECT_FLOAT_EQ( t2->gradp().get()[0], 1.0f );

    std::shared_ptr<Tensor> grad_output = std::make_shared<Tensor>(5.0f);
    t3->backward(grad_output);
    EXPECT_FLOAT_EQ( t1->gradp().get()[0], 6.0f );
    EXPECT_FLOAT_EQ( t2->gradp().get()[0], 6.0f );
    

    // 2D + 2D
    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}}, true);
    std::shared_ptr<Tensor> t5 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{5.0f, 6.0f}, {7.0f, 8.0f}}, true);
    std::shared_ptr<Tensor> t6 = (*t4) + t5;
    t6->backward(std::make_shared<Tensor>(std::vector<std::vector<float>>{{2.0f, 2.0f}, {3.0f, 3.0f}}));
    EXPECT_FLOAT_EQ( t4->gradp().get()[0], 2.0f );
    EXPECT_FLOAT_EQ( t4->gradp().get()[1], 2.0f );
    EXPECT_FLOAT_EQ( t4->gradp().get()[2], 3.0f );
    EXPECT_FLOAT_EQ( t4->gradp().get()[3], 3.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[0], 2.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[1], 2.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[2], 3.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[3], 3.0f );

    
    // 3D + 3D
    std::shared_ptr<Tensor> t7 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}}, true);
    std::shared_ptr<Tensor> t8 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{9.0f, 10.0f}, {11.0f, 12.0f}}, {{13.0f, 14.0f}, {15.0f, 16.0f}}}, true);
    std::shared_ptr<Tensor> t9 = (*t7) + t8;
    t9->backward(std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{1.0f, 1.0f}, {1.0f, 1.0f}}, {{1.0f, 1.0f}, {1.0f, 1.0f}}}));
    for (std::size_t i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ( t9->gradp().get()[i], 1.0f );
        EXPECT_FLOAT_EQ( t7->gradp().get()[i], 1.0f );
        EXPECT_FLOAT_EQ( t8->gradp().get()[i], 1.0f );
    }


    // Test broadcasting
    std::shared_ptr<Tensor> t10 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}}, true);
    std::shared_ptr<Tensor> t11 = std::make_shared<Tensor>(std::vector<float>{5.0f, 6.0f}, true);
    std::shared_ptr<Tensor> t12 = (*t10) + t11;
    t12->backward(std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 1.0f}, {1.0f, 1.0f}}));
    EXPECT_FLOAT_EQ( t10->gradp().get()[0], 1.0f );
    EXPECT_FLOAT_EQ( t10->gradp().get()[1], 1.0f );
    EXPECT_FLOAT_EQ( t10->gradp().get()[2], 1.0f );
    EXPECT_FLOAT_EQ( t10->gradp().get()[3], 1.0f );
    EXPECT_FLOAT_EQ( t11->gradp().get()[0], 2.0f );
    EXPECT_FLOAT_EQ( t11->gradp().get()[1], 2.0f );

    // (3, 1, 2) + (2, 2) -> (3, 2, 2)
    std::shared_ptr<Tensor> t13 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{1.0f, 2.0f}}, {{3.0f, 4.0f}}, {{5.0f, 6.0f}}}, true);
    std::shared_ptr<Tensor> t14 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{7.0f, 8.0f}, {9.0f, 10.0f}}, true);
    std::shared_ptr<Tensor> t15 = (*t13) + t14;
    t15->backward(std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{1.0f, 1.0f}, {1.0f, 1.0f}}, {{1.0f, 1.0f}, {1.0f, 1.0f}}, {{1.0f, 1.0f}, {1.0f, 1.0f}}}));
    for (std::size_t i = 0; i < t15->size(); i++) {
        EXPECT_FLOAT_EQ( t15->gradp().get()[i], 1.0f );
    }
    for (std::size_t i = 0; i < t13->size(); i++) {
        EXPECT_FLOAT_EQ( t13->gradp().get()[i], 2.0f );
    }
    for (std::size_t i = 0; i < t14->size(); i++) {
        EXPECT_FLOAT_EQ( t14->gradp().get()[i], 3.0f );
    }

    // multiple additions
    std::shared_ptr<Tensor> t19 = std::make_shared<Tensor>(2.0f, true);
    std::shared_ptr<Tensor> t20 = std::make_shared<Tensor>(3.0f, true);
    std::shared_ptr<Tensor> t21 = (*t19) + t20;
    std::shared_ptr<Tensor> t22 = std::make_shared<Tensor>(4.0f, true);
    std::shared_ptr<Tensor> t23 = (*t21) + t22;
    t23->backward();
    EXPECT_FLOAT_EQ( t19->gradp().get()[0], 1.0f );
    EXPECT_FLOAT_EQ( t20->gradp().get()[0], 1.0f );
    EXPECT_FLOAT_EQ( t21->gradp().get()[0], 1.0f );
    EXPECT_FLOAT_EQ( t22->gradp().get()[0], 1.0f );
    
}

TEST(TensorTest, BackpropIndexing)
{
    
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<float>{{1.0f, 2.0f}}, true);
    std::shared_ptr<Tensor> t2 = (*t1)[0];
    t2->backward();
    EXPECT_FLOAT_EQ( t1->gradp().get()[0], 1.0f );
    EXPECT_FLOAT_EQ( t1->gradp().get()[1], 0.0f );

    std::shared_ptr<Tensor> t3 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}}, true);
    std::shared_ptr<Tensor> t4 = (*t3)[1];
    t4->backward(std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 1.0f}, {1.0f, 1.0f}}));
    EXPECT_FLOAT_EQ( t3->gradp().get()[0], 0.0f );
    EXPECT_FLOAT_EQ( t3->gradp().get()[1], 0.0f );
    EXPECT_FLOAT_EQ( t3->gradp().get()[2], 0.0f );
    EXPECT_FLOAT_EQ( t3->gradp().get()[3], 0.0f );
    EXPECT_FLOAT_EQ( t3->gradp().get()[4], 1.0f );
    EXPECT_FLOAT_EQ( t3->gradp().get()[5], 1.0f );
    EXPECT_FLOAT_EQ( t3->gradp().get()[6], 1.0f );
    EXPECT_FLOAT_EQ( t3->gradp().get()[7], 1.0f );

    std::shared_ptr<Tensor> t5 = std::make_shared<Tensor>(Tensor::fill(2.0f, {2, 2, 2}));
    t5->set_require_grad(true);
    std::shared_ptr<Tensor> t6 = (*t5)[0, 1];
    t6->backward(std::make_shared<Tensor>(std::vector<float>{1.0f, 1.0f}));
    EXPECT_FLOAT_EQ( t5->gradp().get()[0], 0.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[1], 0.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[2], 1.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[3], 1.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[4], 0.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[5], 0.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[6], 0.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[7], 0.0f );
    

    std::shared_ptr<Tensor> t7 = std::make_shared<Tensor>(Tensor::fill(2.0f, {2, 2, 2}));
    t7->set_require_grad(true);
    std::shared_ptr<Tensor> t8 = (*t7)[1];
    std::shared_ptr<Tensor> t9 = (*t8)[0];

    t9->backward(std::make_shared<Tensor>(std::vector<float>{1.0f, 1.0f}));
    EXPECT_EQ( (t7->shape()), (std::vector<std::size_t>{2, 2, 2}) );
    EXPECT_EQ( (t8->shape()), (std::vector<std::size_t>{2, 2}) );
    EXPECT_EQ( (t9->shape()), (std::vector<std::size_t>{2}) );

    EXPECT_EQ( t7->size(), 8 );
    EXPECT_EQ( t8->size(), 4 );
    EXPECT_EQ( t9->size(), 2 );

    EXPECT_EQ( t7->offset(), 0 );
    EXPECT_EQ( t8->offset(), 4 );
    EXPECT_EQ( t9->offset(), 4 );

    EXPECT_FLOAT_EQ( t9->gradp().get()[0], 1.0f );
    EXPECT_FLOAT_EQ( t9->gradp().get()[1], 1.0f );

    EXPECT_FLOAT_EQ( t8->gradp().get()[0], 1.0f );
    EXPECT_FLOAT_EQ( t8->gradp().get()[1], 1.0f );
    EXPECT_FLOAT_EQ( t8->gradp().get()[2], 0.0f );
    EXPECT_FLOAT_EQ( t8->gradp().get()[3], 0.0f );

    EXPECT_FLOAT_EQ( t7->gradp().get()[0], 0.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[1], 0.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[2], 0.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[3], 0.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[4], 1.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[5], 1.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[6], 0.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[7], 0.0f );
}

TEST(TensorTest, BackpropMul)
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(2.0f, true);
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(3.0f, true);
    std::shared_ptr<Tensor> t3 = (*t1) * t2;
    t3->backward();

    EXPECT_FLOAT_EQ( t1->gradp().get()[0], 3.0f );
    EXPECT_FLOAT_EQ( t2->gradp().get()[0], 2.0f );

    std::shared_ptr<Tensor> grad_output = std::make_shared<Tensor>(5.0f);
    t3->backward(grad_output);
    EXPECT_FLOAT_EQ( t1->gradp().get()[0], 18.0f );
    EXPECT_FLOAT_EQ( t2->gradp().get()[0], 12.0f );

    // 2D * 2D
    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}}, true);
    std::shared_ptr<Tensor> t5 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{5.0f, 6.0f}, {7.0f, 8.0f}}, true);
    std::shared_ptr<Tensor> t6 = (*t4) * t5;
    t6->backward(std::make_shared<Tensor>(std::vector<std::vector<float>>{{2.0f, 2.0f}, {3.0f, 3.0f}}));
    EXPECT_FLOAT_EQ( t4->gradp().get()[0], 10.0f );
    EXPECT_FLOAT_EQ( t4->gradp().get()[1], 12.0f );
    EXPECT_FLOAT_EQ( t4->gradp().get()[2], 21.0f );
    EXPECT_FLOAT_EQ( t4->gradp().get()[3], 24.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[0], 2.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[1], 4.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[2], 9.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[3], 12.0f );

    // 3D * 3D
    std::shared_ptr<Tensor> t7 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}}, true);
    std::shared_ptr<Tensor> t8 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{9.0f, 10.0f}, {11.0f, 12.0f}}, {{13.0f, 14.0f}, {15.0f, 16.0f}}}, true);
    std::shared_ptr<Tensor> t9 = (*t7) * t8;
    t9->backward(std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{1.0f, 1.0f}, {1.0f, 1.0f}}, {{1.0f, 1.0f}, {1.0f, 1.0f}}}));
    EXPECT_FLOAT_EQ( t7->gradp().get()[0], 9.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[1], 10.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[2], 11.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[3], 12.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[4], 13.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[5], 14.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[6], 15.0f );
    EXPECT_FLOAT_EQ( t7->gradp().get()[7], 16.0f );
    EXPECT_FLOAT_EQ( t8->gradp().get()[0], 1.0f );
    EXPECT_FLOAT_EQ( t8->gradp().get()[1], 2.0f );
    EXPECT_FLOAT_EQ( t8->gradp().get()[2], 3.0f );
    EXPECT_FLOAT_EQ( t8->gradp().get()[3], 4.0f );
    EXPECT_FLOAT_EQ( t8->gradp().get()[4], 5.0f );
    EXPECT_FLOAT_EQ( t8->gradp().get()[5], 6.0f );
    EXPECT_FLOAT_EQ( t8->gradp().get()[6], 7.0f );
    EXPECT_FLOAT_EQ( t8->gradp().get()[7], 8.0f );

    // 1D * 2D
    std::shared_ptr<Tensor> t10 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f}, true);
    std::shared_ptr<Tensor> t11 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{3.0f, 4.0f}, {5.0f, 6.0f}}, true);
    std::shared_ptr<Tensor> t12 = (*t10) * t11;
    t12->backward(std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 1.0f}, {1.0f, 1.0f}}));
    EXPECT_FLOAT_EQ( t10->gradp().get()[0], 8.0f );
    EXPECT_FLOAT_EQ( t10->gradp().get()[1], 10.0f );
    EXPECT_FLOAT_EQ( t11->gradp().get()[0], 1.0f );
    EXPECT_FLOAT_EQ( t11->gradp().get()[1], 2.0f );
    EXPECT_FLOAT_EQ( t11->gradp().get()[2], 1.0f );
    EXPECT_FLOAT_EQ( t11->gradp().get()[3], 2.0f );

    // 3D * 2D
    std::shared_ptr<Tensor> t13 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}}, true);
    std::shared_ptr<Tensor> t14 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{9.0f, 10.0f}, {11.0f, 12.0f}}, true);
    std::shared_ptr<Tensor> t15 = (*t13) * t14;
    t15->backward(std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{1.0f, 1.0f}, {1.0f, 1.0f}}, {{1.0f, 1.0f}, {1.0f, 1.0f}}}));
    EXPECT_FLOAT_EQ( t13->gradp().get()[0], 9.0f );
    EXPECT_FLOAT_EQ( t13->gradp().get()[1], 10.0f );
    EXPECT_FLOAT_EQ( t13->gradp().get()[2], 11.0f );
    EXPECT_FLOAT_EQ( t13->gradp().get()[3], 12.0f );
    EXPECT_FLOAT_EQ( t13->gradp().get()[4], 9.0f );
    EXPECT_FLOAT_EQ( t13->gradp().get()[5], 10.0f );
    EXPECT_FLOAT_EQ( t13->gradp().get()[6], 11.0f );
    EXPECT_FLOAT_EQ( t13->gradp().get()[7], 12.0f );
    EXPECT_FLOAT_EQ( t14->gradp().get()[0], 6.0f );
    EXPECT_FLOAT_EQ( t14->gradp().get()[1], 8.0f );
    EXPECT_FLOAT_EQ( t14->gradp().get()[2], 10.0f );
    EXPECT_FLOAT_EQ( t14->gradp().get()[3], 12.0f );
}

TEST(TensorTest, BackpropMatmul)
{
    // 1D @ 1D
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f}, true);
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(std::vector<float>{4.0f, 5.0f, 6.0f}, true);
    std::shared_ptr<Tensor> t3 = t1->matmul(t2);
    t3->backward();
    EXPECT_FLOAT_EQ( t1->gradp().get()[0], 4.0f );
    EXPECT_FLOAT_EQ( t1->gradp().get()[1], 5.0f );
    EXPECT_FLOAT_EQ( t1->gradp().get()[2], 6.0f );
    EXPECT_FLOAT_EQ( t2->gradp().get()[0], 1.0f );
    EXPECT_FLOAT_EQ( t2->gradp().get()[1], 2.0f );
    EXPECT_FLOAT_EQ( t2->gradp().get()[2], 3.0f );

    // 2D @ 2D
    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}}, true);
    std::shared_ptr<Tensor> t5 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{5.0f, 6.0f}, {7.0f, 8.0f}}, true);
    std::shared_ptr<Tensor> t6 = t4->matmul(t5);
    t6->backward(std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 1.0f}, {1.0f, 1.0f}}));
    EXPECT_FLOAT_EQ( t4->gradp().get()[0], 11.0f );
    EXPECT_FLOAT_EQ( t4->gradp().get()[1], 15.0f );
    EXPECT_FLOAT_EQ( t4->gradp().get()[2], 11.0f );
    EXPECT_FLOAT_EQ( t4->gradp().get()[3], 15.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[0], 4.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[1], 4.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[2], 6.0f );
    EXPECT_FLOAT_EQ( t5->gradp().get()[3], 6.0f );

    // 3D @ 2D
    std::shared_ptr<Tensor> t19 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 2.0f}, {3.0f, 4.0f}},
        {{5.0f, 6.0f}, {7.0f, 8.0f}}
    }, true);
    std::shared_ptr<Tensor> t20 = std::make_shared<Tensor>(std::vector<std::vector<float>>{
        {9.0f, 10.0f},
        {11.0f, 12.0f}
    }, true);
    std::shared_ptr<Tensor> t21 = t19->matmul(t20);
    t21->backward(std::make_shared<Tensor>(Tensor::fill(1.0f, {2, 2, 2})));
    EXPECT_FLOAT_EQ( t19->gradp().get()[0], 19.0f );
    EXPECT_FLOAT_EQ( t19->gradp().get()[1], 23.0f );
    EXPECT_FLOAT_EQ( t19->gradp().get()[2], 19.0f );
    EXPECT_FLOAT_EQ( t19->gradp().get()[3], 23.0f );
    EXPECT_FLOAT_EQ( t20->gradp().get()[0], 16.0f );
    EXPECT_FLOAT_EQ( t20->gradp().get()[1], 16.0f );
    EXPECT_FLOAT_EQ( t20->gradp().get()[2], 20.0f );
    EXPECT_FLOAT_EQ( t20->gradp().get()[3], 20.0f );

    // 3D @ 3D
    std::shared_ptr<Tensor> t25 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 2.0f}, {3.0f, 4.0f}},
        {{5.0f, 6.0f}, {7.0f, 8.0f}}
    }, true);
    std::shared_ptr<Tensor> t26 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{9.0f, 10.0f}, {11.0f, 12.0f}},
        {{13.0f, 14.0f}, {15.0f, 16.0f}}
    }, true);
    std::shared_ptr<Tensor> t27 = t25->matmul(t26);
    t27->backward(std::make_shared<Tensor>(Tensor::fill(1.0f, {2, 2, 2})));
    EXPECT_FLOAT_EQ( t25->gradp().get()[0], 19.0f );
    EXPECT_FLOAT_EQ( t25->gradp().get()[1], 23.0f );
    EXPECT_FLOAT_EQ( t25->gradp().get()[2], 19.0f );
    EXPECT_FLOAT_EQ( t25->gradp().get()[3], 23.0f );
    EXPECT_FLOAT_EQ( t25->gradp().get()[4], 27.0f );
    EXPECT_FLOAT_EQ( t25->gradp().get()[5], 31.0f );
    EXPECT_FLOAT_EQ( t25->gradp().get()[6], 27.0f );
    EXPECT_FLOAT_EQ( t25->gradp().get()[7], 31.0f );

    EXPECT_FLOAT_EQ( t26->gradp().get()[0], 4.0f );
    EXPECT_FLOAT_EQ( t26->gradp().get()[1], 4.0f );
    EXPECT_FLOAT_EQ( t26->gradp().get()[2], 6.0f );
    EXPECT_FLOAT_EQ( t26->gradp().get()[3], 6.0f );
    EXPECT_FLOAT_EQ( t26->gradp().get()[4], 12.0f );
    EXPECT_FLOAT_EQ( t26->gradp().get()[5], 12.0f );
    EXPECT_FLOAT_EQ( t26->gradp().get()[6], 14.0f );
    EXPECT_FLOAT_EQ( t26->gradp().get()[7], 14.0f );

    // 3D @ 1D
    std::shared_ptr<Tensor> t28 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{
        {{1.0f, 2.0f}, {3.0f, 4.0f}},
        {{5.0f, 6.0f}, {7.0f, 8.0f}}
    }, true);
    std::shared_ptr<Tensor> t29 = std::make_shared<Tensor>(std::vector<float>{9.0f, 10.0f}, true);
    std::shared_ptr<Tensor> t30 = t28->matmul(t29);
    t30->backward(std::make_shared<Tensor>(Tensor::fill(1.0f, {2, 2})));
    EXPECT_FLOAT_EQ( t28->gradp().get()[0], 9.0f );
    EXPECT_FLOAT_EQ( t28->gradp().get()[1], 10.0f );
    EXPECT_FLOAT_EQ( t28->gradp().get()[2], 9.0f );
    EXPECT_FLOAT_EQ( t28->gradp().get()[3], 10.0f );
    EXPECT_FLOAT_EQ( t28->gradp().get()[4], 9.0f );
    EXPECT_FLOAT_EQ( t28->gradp().get()[5], 10.0f );
    EXPECT_FLOAT_EQ( t28->gradp().get()[6], 9.0f );
    EXPECT_FLOAT_EQ( t28->gradp().get()[7], 10.0f );
    EXPECT_FLOAT_EQ( t29->gradp().get()[0], 16.0f );
    EXPECT_FLOAT_EQ( t29->gradp().get()[1], 20.0f );

    // 4D @ 4D
    std::shared_ptr<Tensor> t31 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<std::vector<float>>>>{
        {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}},
        {{{9.0f, 10.0f}, {11.0f, 12.0f}}, {{13.0f, 14.0f}, {15.0f, 16.0f}}}
    }, true);
    std::shared_ptr<Tensor> t32 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<std::vector<float>>>>{
        {{{17.0f, 18.0f}, {19.0f, 20.0f}}, {{21.0f, 22.0f}, {23.0f, 24.0f}}},
        {{{25.0f, 26.0f}, {27.0f, 28.0f}}, {{29.0f, 30.0f}, {31.0f, 32.0f}}}
    }, true);
    std::shared_ptr<Tensor> t33 = t31->matmul(t32);
    t33->backward(std::make_shared<Tensor>(Tensor::fill(1.0f, {2, 2, 2, 2})));
    EXPECT_FLOAT_EQ( t31->gradp().get()[0], 35.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[1], 39.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[2], 35.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[3], 39.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[4], 43.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[5], 47.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[6], 43.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[7], 47.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[8], 51.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[9], 55.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[10], 51.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[11], 55.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[12], 59.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[13], 63.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[14], 59.0f );
    EXPECT_FLOAT_EQ( t31->gradp().get()[15], 63.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[0], 4.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[1], 4.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[2], 6.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[3], 6.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[4], 12.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[5], 12.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[6], 14.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[7], 14.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[8], 20.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[9], 20.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[10], 22.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[11], 22.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[12], 28.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[13], 28.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[14], 30.0f );
    EXPECT_FLOAT_EQ( t32->gradp().get()[15], 30.0f );
}



