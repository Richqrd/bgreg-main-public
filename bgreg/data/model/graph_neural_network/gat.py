from spektral.layers import ECCConv, GlobalAvgPool, GATConv
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class Net(Model):
    def __init__(self, fltrs_out=64, l2_reg=1e-3, dropout_rate=0.5, classify="binary"):
        super().__init__()
        self.conv1 = GATConv(fltrs_out, activation="relu", kernel_regularizer=l2(l2_reg),
                             attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.conv2 = GATConv(fltrs_out, activation="relu", kernel_regularizer=l2(l2_reg),
                             attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.conv3 = GATConv(fltrs_out, activation="relu", kernel_regularizer=l2(l2_reg),
                             attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.conv4 = GATConv(fltrs_out, activation="relu", kernel_regularizer=l2(l2_reg),
                             attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.conv5 = GATConv(fltrs_out, activation="relu", kernel_regularizer=l2(l2_reg),
                             attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.flatten = GlobalAvgPool()
        self.fc = Dense(32, "relu", kernel_regularizer=l2(l2_reg))
        # self.dropout = Dropout(dropout_rate)
        if classify == "binary":
            self.out = Dense(1, "sigmoid", kernel_regularizer=l2(l2_reg))
        elif classify == "multi":
            self.out = Dense(3, "softmax", kernel_regularizer=l2(l2_reg))


    def call(self, inputs, training):
        A_in, X_in, E_in = inputs

        embeddings = []

        print('Node features shape')
        print(X_in.shape)
        print('Edge features shape')
        print(E_in.shape)

        embeddings.append(X_in)
        x, attn = self.conv1([X_in, A_in])
        embeddings.append(x)
        x, _ = self.conv2([x, A_in])
        embeddings.append(x)
        x, _ = self.conv3([x, A_in])
        embeddings.append(x)
        x, _ = self.conv4([x, A_in])
        embeddings.append(x)
        x, _ = self.conv5([x, A_in])

        embeddings.append(x)
        x = self.flatten(x)
        embeddings.append(x)
        x = self.fc(x)
        embeddings.append(x)
        # x = self.dropout(x)
        output = self.out(x)
        embeddings.append(output)

        # return attention only when not training
        if training:
            return output

        else:
            return output, attn, embeddings
