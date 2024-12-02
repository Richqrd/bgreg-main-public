from spektral.layers import ECCConv, GlobalAvgPool, GATConv
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class Net(Model):
    def __init__(self, fltrs_out=64, l2_reg=1e-3, dropout_rate=0.5, classify="binary"):
        super().__init__()

        self.test_cnn = Conv1D(filters=16, kernel_size=5, activation="relu")
        self.test_cnn1 = Conv1D(filters=8, kernel_size=3, activation="relu")
        self.test_pool = MaxPooling1D(pool_size=3)
        self.test_flat = Flatten()
        self.test_fc = Dense(8, "relu", kernel_regularizer=l2(l2_reg))

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

        # Process node features

        unstacked = tf.unstack(X_in, axis=1)
        stacks = []
        for stack in unstacked:
            processed_stack = tf.expand_dims(stack, axis=-1)
            # print(processed_stack.shape)
            processed_stack = self.test_cnn(processed_stack)
            processed_stack = self.test_cnn1(processed_stack)
            processed_stack = self.test_pool(processed_stack)
            processed_stack = self.test_flat(processed_stack)
            # print(processed_stack.shape)
            processed_stack = self.test_fc(processed_stack)
            # print(processed_stack.shape)
            stacks.append(processed_stack)

        x = tf.stack(stacks, axis=1)

        # x = self.test_fc(X_in) # becomes shape batches * electrodes * compute features

        print('New node features shape')
        print(x.shape)


        x = self.conv1([x, A_in, E_in])
        embeddings.append(x)
        x, attn = self.conv2([x, A_in])
        embeddings.append(x)
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.dropout(x)
        output = self.out(x)

        # return attention only when not training
        if training:
            return output

        else:
            return output, attn, embeddings
