#include"model.h"

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define Data_Length 579

typedef struct _Data
{
    Mat *input;
    double output_hat;
}Data;

int main()
{
    FILE *fp = fopen("579_norm.txt", "r");
    Data data[Data_Length];
    Mat *state_t[Data_Length], *temp;
    double output[Data_Length], loss, total_loss;
    RNN *model = create_model();
    RNN_Gradient *gradient = create_gradient();
    int index, epoch;
    srand(time(NULL));

    for(index = 0; index < Data_Length; index++)
    {
        data[index].input = new_matrix(1, 2);
        state_t[index] = new_matrix(2, 1);
        fscanf(fp, "%lf", &data[index].input->element[0]);
        fscanf(fp, "%lf", &data[index].input->element[1]);
        fscanf(fp, "%lf", &data[index].output_hat);
    }
    fclose(fp);

    fp = fopen("loss.csv", "w");
    for(epoch = 0; epoch < 4000; epoch++)
    {
        total_loss = 0.0;
        init_matrix(model->state);
        for(index = 0; index < Data_Length; index++)
        {
            temp = transpose(model->state);
            copy_matrix(state_t[index], temp);
            delete_matrix(temp);
            forward(model, data[index].input);
            output[index] = model->output;
        }
        temp = transpose(model->w_sh);
        copy_matrix(gradient->w_sh_t, temp);
        delete_matrix(temp);
        init_matrix(gradient->g_ih);
        init_matrix(gradient->g_ho);
        init_matrix(gradient->g_sh);
        init_matrix(model->state);
        for(index = Data_Length - 1; index >= 0; index--)
        {
            loss = data[index].output_hat - output[index];
            total_loss += loss * loss / 2;
            temp = transpose(data[index].input);
            copy_matrix(gradient->input_t, temp);
            delete_matrix(temp);
            copy_matrix(gradient->state_t, state_t[index]);
            backward(model, gradient, loss);
        }
        fprintf(fp, "%d,%lf\n", epoch, total_loss / Data_Length);
        update(model, gradient, 1e-2);
    }
    fclose(fp);

    fp = fopen("output.csv", "w");
    init_matrix(model->state);
    for(index = 0; index < Data_Length; index++)
    {
        forward(model, data[index].input);
        fprintf(fp, "%d,%lf\n", index, model->output);
    }
    fclose(fp);

    return 0;
}