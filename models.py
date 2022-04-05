
import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x,self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        size = dataset.x.shape[0]
        correct = 0
        accuracy = 0
        while (accuracy != 1):
            for x,y in dataset.iterate_once(batch_size):
                prediction = self.get_prediction(x)
                if (prediction != nn.as_scalar(y)):
                    self.w.update(x,-prediction)
                else:
                    correct+=1
            accuracy = correct/size
            correct = 0

            


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 10
        self.layer_size = 50
        self.layer_size2= 50
        self.learning_rate = -0.001
        self.w_1 = nn.Parameter(1,self.layer_size)
        self.w_1_2 = nn.Parameter(self.layer_size,self.layer_size2)
        self.w_2 = nn.Parameter(self.layer_size2,1)
        self.b_1 = nn.Parameter(1,self.layer_size)
        self.b_1_2 = nn.Parameter(1,self.layer_size2)
        self.b_2 = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # batchsize x 1  *  1 x layersize = batchsize x layersize
        xw_1 = nn.Linear(x,self.w_1)
        #batchsize x layersize + 1 x layersize = batchsize x layersize
        xw_1_b1 = nn.AddBias(xw_1,self.b_1)
        # batchsize x layersize
        relu = nn.ReLU(xw_1_b1)
        # batchsize x layersize * layersize x layersize2 = batchsize x layersize2
        relu_x_w_1_2 = nn.Linear(relu,self.w_1_2)
        #
        relu_x_w_1_2_b_1_2 = nn.AddBias(relu_x_w_1_2,self.b_1_2)
        second_relu = nn.ReLU(relu_x_w_1_2_b_1_2)
        # batchsize x layersize + layersize x 1 = batchsize x 1
        reluw_2 = nn.Linear(second_relu,self.w_2)
        return nn.AddBias(reluw_2,self.b_2)
    

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss_value = float("inf")
        while (loss_value > 0.0001):
            for x , y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                grad_w_1 , grad_w_2, grad_w_1_2, grad_b_1, grad_b_2, grad_b_1_2 = nn.gradients(loss,[self.w_1,self.w_2,self.w_1_2,self.b_1,self.b_2,self.b_1_2])
                self.w_1.update(grad_w_1,self.learning_rate)
                self.w_2.update(grad_w_2,self.learning_rate)
                self.w_1_2.update(grad_w_1_2,self.learning_rate)
                self.b_1.update(grad_b_1,self.learning_rate)
                self.b_2.update(grad_b_2,self.learning_rate)
                self.b_1_2.update(grad_b_1_2,self.learning_rate)
                loss_value = nn.as_scalar(loss)


            

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(784,250)
        self.b1 = nn.Parameter(1,250)
        self.W2 = nn.Parameter(250,150)
        self.b2 = nn.Parameter(1,150)
        self.W3 = nn.Parameter(150,10)
        self.b3 = nn.Parameter(1,10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        Z1 = nn.AddBias(nn.Linear(x,self.W1),self.b1)
        A1 = nn.ReLU(Z1)
        Z2 = nn.AddBias(nn.Linear(A1,self.W2),self.b2)
        A2 = nn.ReLU(Z2)
        Z3 = nn.AddBias(nn.Linear(A2,self.W3),self.b3)
        return Z3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #loss_value = float("inf")

        acc = -float('inf')
        while acc<0.976:
            for x,y in dataset.iterate_once(100): 
                grad_wrt_W1, grad_wrt_b1, grad_wrt_W2, grad_wrt_b2, grad_wrt_W3, grad_wrt_b3  = nn.gradients(self.get_loss(x,y), [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
                self.W1.update(grad_wrt_W1, -0.34)
                self.b1.update(grad_wrt_b1, -0.34)
                self.W2.update(grad_wrt_W2, -0.34)
                self.b2.update(grad_wrt_b2, -0.34)
                self.W3.update(grad_wrt_W3, -0.34)
                self.b3.update(grad_wrt_b3, -0.34)
            acc = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.dimensionality = 80
        self.firstLayerSize = 100
        self.secondLayerSize = 100
        self.W1 = nn.Parameter(self.num_chars,self.firstLayerSize)
        self.W2 = nn.Parameter(self.firstLayerSize,self.secondLayerSize)
        self.W3 = nn.Parameter(self.secondLayerSize,self.dimensionality)
        self.b1 = nn.Parameter(1,self.firstLayerSize)
        self.b2 = nn.Parameter(1,self.secondLayerSize)
        self.b3 = nn.Parameter(1,self.dimensionality)
        self.W1h = nn.Parameter(self.dimensionality,self.firstLayerSize)
        self.W2h = nn.Parameter(self.firstLayerSize,self.secondLayerSize)
        self.W3h = nn.Parameter(self.secondLayerSize,self.dimensionality)
        self.b1h = nn.Parameter(1,self.firstLayerSize)
        self.b2h = nn.Parameter(1,self.secondLayerSize)
        self.b3h = nn.Parameter(1,self.dimensionality)
        self.Wfinal =nn.Parameter(self.dimensionality,5)
        self.bfinal = nn.Parameter(1,5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = None
        for i in range(len(xs)):
            if not i:
                Z0 = nn.Linear(xs[i],self.W1)
                A0 = nn.ReLU(nn.AddBias(Z0,self.b1))
                Z1 = nn.Linear(A0,self.W2)
                A1 = nn.ReLU(nn.AddBias(Z1,self.b2))
                h = nn.AddBias(nn.Linear(A1,self.W3),self.b3)
            else:
                A1 = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(xs[i],self.W1),nn.Linear(h,self.W1h)),self.b1h))
                Z2 = nn.AddBias(nn.Linear(A1,self.W2h),self.b2h)
                A2 = nn.ReLU(Z2)
                h = nn.AddBias(nn.Linear(A2,self.W3h),self.b3h)

        return nn.AddBias(nn.Linear(h,self.Wfinal),self.bfinal)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        acc = -float('inf')
        for i in range(20):
        #while (acc <0.9):
            for x,y in dataset.iterate_once(100): 
                grad_wrt_W1, grad_wrt_b1, grad_wrt_W2, grad_wrt_b2,grad_wrt_W3, grad_wrt_b3, grad_wrt_W1h, grad_wrt_W2h, grad_wrt_W3h,  grad_wrt_b1h, grad_wrt_b2h, grad_wrt_b3h, grad_wrt_Wfinal, grad_wrt_bfinal  = nn.gradients(self.get_loss(x,y), [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W1h,self.W2h,self.W3h, self.b1h,self.b2h,self.b3h,self.Wfinal,self.bfinal])
                self.W1.update(grad_wrt_W1, -0.05)
                self.b1.update(grad_wrt_b1, -0.05)
                self.W2.update(grad_wrt_W2, -0.05)
                self.b2.update(grad_wrt_b2, -0.05)
                self.W3.update(grad_wrt_W3, -0.05)
                self.b3.update(grad_wrt_b3, -0.05)
                self.W1h.update(grad_wrt_W1h, -0.05)
                self.W2h.update(grad_wrt_W2h, -0.05)
                self.W3h.update(grad_wrt_W3h, -0.05)
                self.b1h.update(grad_wrt_b1h, -0.05)
                self.b2h.update(grad_wrt_b2h, -0.05)
                self.b3h.update(grad_wrt_b3h, -0.05)
                self.Wfinal.update(grad_wrt_Wfinal, -0.05)
                self.bfinal.update(grad_wrt_bfinal, -0.05)
            acc = dataset.get_validation_accuracy()
