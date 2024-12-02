from spektral.layers import ECCConv, GlobalAvgPool, GATConv
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class Net(Model):
    def __init__(self, fltrs_out=64, l2_reg=1e-3, dropout_rate=0.5, classify="binary"):
        super().__init__()
        self.conv1 = ECCConv(fltrs_out, kernel_network=[32], activation="relu", kernel_regularizer=l2(l2_reg))
        self.conv2 = GATConv(fltrs_out, activation="relu", kernel_regularizer=l2(l2_reg),
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


        x = self.conv1([X_in, A_in, E_in])
        embeddings.append(x)
        # x, attn = self.conv2([X_in, A_in])
        embeddings.append(x)
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.dropout(x)
        output = self.out(x)

        # return attention only when not training
        if training:
            return output

        else:
            return output, None, embeddings