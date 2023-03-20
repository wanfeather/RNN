#include"nn.h"

#include<math.h>

void Sigmoid(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = sigmoid(input->element[index]);
}

void Tanh(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = tanh(input->element[index]);
}

void Relu(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = relu(input->element[index]);
}

void Softmax(Mat *input)
{
    int index;
    double sum = 0.0;

    for(index = 0; index < input->row * input->col; index++)
    {
        input->element[index] = exp(input->element[index]);
        sum += input->element[index];
    }
    for(index = 0; index < input->row * input->col; index++)
        input->element[index] /= sum;
}

void Sigmoid_gradient(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = sigmoid_gradient(input->element[index]);
}

void Tanh_gradient(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = tanh_gradient(input->element[index]);
}

void Relu_gradient(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = relu_gradient(input->element[index]);
}
/*
double Cross_Entropy(Mat *output, int class)
{
    double loss = -log(output->element[class]--);

    return loss;
}
*/