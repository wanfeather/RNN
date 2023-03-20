#include"model.h"

#include<stdlib.h>
#include<math.h>

RNN *create_model(void)
{
    RNN *model = (RNN *)malloc(sizeof(RNN));
    int index_1, index_2;
    double k;

    model->input = new_matrix(1, 2);
    model->hidden = new_matrix(1, 2);
    model->state = new_matrix(1, 2);
    model->w_ih = new_matrix(2, 2);
    model->w_ho = new_matrix(2, 1);
    model->w_sh = new_matrix(2, 2);

    k = 1.0 / sqrt(2.0);
    for(index_1 = 0; index_1 < 2; index_1++)
    {
        model->w_ho->element[index_1] = xavier_init(k);
        for(index_2 = 0; index_2 < 2; index_2++)
        {
            model->w_ih->element[index_1 * 2 + index_2] = xavier_init(k);
            model->w_sh->element[index_1 * 2 + index_2] = xavier_init(k);
        }
    }

    return model;
}

RNN_Gradient *create_gradient(void)
{
    RNN_Gradient *gradient = (RNN_Gradient *)malloc(sizeof(RNN_Gradient));

    gradient->input_t = new_matrix(2, 1);
    gradient->w_sh_t = new_matrix(2, 2);
    gradient->state_t = new_matrix(2, 1);
    gradient->g_ih = new_matrix(2, 2);
    gradient->g_ho = new_matrix(2, 1);
    gradient->g_sh = new_matrix(2, 2);

    return gradient;
}

double xavier_init(double k)
{
    double init_w = 2 * k * rand() / RAND_MAX - k;

    return init_w;
}

void forward(RNN *model, Mat *data)
{
    Mat *temp_1, *temp_2, *temp_3;

    copy_matrix(model->input, data);
    temp_1 = matrix_product(model->input, model->w_ih);
    temp_2 = matrix_product(model->state, model->w_sh);
    temp_3 = matrix_addtion(temp_1, temp_2);
    copy_matrix(model->hidden, temp_3);
    copy_matrix(model->state, temp_3);
    Sigmoid(model->hidden);

    delete_matrix(temp_1);
    delete_matrix(temp_2);
    delete_matrix(temp_3);

    temp_1 = matrix_product(model->hidden, model->w_ho);
    model->output = temp_1->element[0];
    delete_matrix(temp_1);
}

void backward(RNN *model, RNN_Gradient *gradient, double loss)
{
    Mat *temp_1, *temp_2, *temp_3;

    temp_1 = scale_product(loss, model->hidden);
    temp_2 = matrix_addtion(gradient->g_ho, temp_1);
    copy_matrix(gradient->g_ho, temp_2);
    delete_matrix(temp_1);
    delete_matrix(temp_2);

    temp_1 = scale_product(loss, model->w_ho);
    temp_2 = matrix_addtion(model->state, temp_1);
    Sigmoid_gradient(model->hidden);
    delete_matrix(temp_1);
    temp_1 = element_product(model->hidden, temp_2);
    delete_matrix(temp_2);
    temp_2 = matrix_product(gradient->input_t, temp_1);
    temp_3 = matrix_addtion(gradient->g_ih, temp_2);
    copy_matrix(gradient->g_ih, temp_3);
    delete_matrix(temp_2);
    delete_matrix(temp_3);
    temp_2 = matrix_product(gradient->state_t, temp_1);
    temp_3 = matrix_addtion(gradient->g_sh, temp_2);
    copy_matrix(gradient->g_sh, temp_3);
    delete_matrix(temp_2);
    delete_matrix(temp_3);
    temp_2 = matrix_product(temp_1, gradient->w_sh_t);
    copy_matrix(model->state, temp_2);
    delete_matrix(temp_1);
    delete_matrix(temp_2);
}

void update(RNN *model, RNN_Gradient *gradient, double lr)
{
    SGD(model->w_ih, gradient->g_ih, lr);
    SGD(model->w_sh, gradient->g_sh, lr);
    SGD(model->w_ho, gradient->g_ho, lr);
}

void SGD(Mat *weight, Mat *gradient, double lr)
{
    int index;

    for(index = 0; index < weight->row * weight->col; index++)
        weight->element[index] += lr * gradient->element[index];
}