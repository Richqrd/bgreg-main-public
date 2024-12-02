from spektral.layers import ECCConv, GlobalAvgPool, GATConv
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class Net(Model):
    def __init__(self, fltrs_out=64, l2_reg=1e-3, dropout_rate=0.5):
        super().__init__()
        self.conv1 = ECCConv(fltrs_out, kernel_network=[32], activation="relu", kernel_regularizer=l2(l2_reg))
        self.conv2 = GATConv(1, activation="sigmoid", kernel_regularizer=l2(l2_reg),
                             attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.dropout = Dropout(dropout_rate)


    def call(self, inputs, training):
        A_in, X_in, E_in = inputs

        embeddings = []

        print('Node features shape')
        print(X_in.shape)
        print('Edge features shape')
        print(E_in.shape)

        embeddings.append(X_in)
        x = self.conv1([X_in, A_in, E_in])
        embeddings.append(x)
        x, attn = self.conv2([x, A_in])
        embeddings.append(x)
        output = x
        embeddings.append(x)

        # return attention only when not training
        if training:
            return output

        else:
            return output, attn, embeddings
