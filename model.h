#ifndef MODEL_H
#define MODEL_H

#include"matrix.h"
#include"nn.h"

typedef struct _RNN
{
    Mat *input, *hidden, *state;
    double output;
    Mat *w_ih, *w_ho, *w_sh;
}RNN;
typedef struct _RNN_Gradient
{
    Mat *input_t, *state_t, *w_sh_t;
    Mat *g_ih, *g_ho, *g_sh;
}RNN_Gradient;

RNN *create_model(void);
RNN_Gradient *create_gradient(void);
double xavier_init(double);
void forward(RNN *, Mat *);
void backward(RNN *, RNN_Gradient *, double);
void update(RNN *, RNN_Gradient *, double);
void SGD(Mat *, Mat *, double);

#endif