"""
Implements a network visualization in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from network_visualization.py!")


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    # Hint: X.grad.data stores the gradients, with shape (N, 3, H, W)            #
    ##############################################################################
    # Replace "pass" statement with your code

    output_scores = model(X) # (N, num_classes)
    y = y.view(-1, 1) # (N, 1)

    correct_score = torch.gather(output_scores, dim=1, index=y).squeeze() # (5,)

    loss = -correct_score.sum() # minimize

    model.zero_grad()
    loss.backward()

    saliency = torch.abs(X.grad.data).max(dim=1).values # (N, 3, H, W) -> max -> (N, H, W)

    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return saliency


def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    """
    Generate an adversarial attack that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    - max_iter: Upper bound on number of iteration to perform
    - verbose: If True, it prints the progress (you can use this flag for debugging)

    Returns:
    - X_adv: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our adversarial attack to the input image, and make it require
    # gradient
    X_adv = X.clone()
    X_adv = X_adv.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate an adversarial attack X_adv that the model will classify    #
    # as the class target_y. You should perform gradient ascent on the score     #
    # of the target class, stopping when the model is fooled.                    #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate an adversarial     #
    # attack in fewer than 100 iterations of gradient ascent.                    #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    # Replace "pass" statement with your code

    for iteration in range(max_iter):
      output_score = model(X_adv) # (1, 1000)
      
      if output_score.max(dim=1).indices.item() == target_y:
        return X_adv

      if verbose:
        print(f'Iteration {iteration}: target score {output_score[:,target_y].item():.3f},\
         max score {output_score.max(dim=1).values.item():.3f}')
      
      loss = output_score[:, target_y].mean() # MAXIMIZE!

      model.zero_grad()
      loss.backward() 

      grad = X_adv.grad / torch.norm(X_adv.grad, p=2) # (1, 3, 244, 244)
      dX = learning_rate * grad

      with torch.no_grad():

        X_adv.data += dX # Gradient Ascent
      
      X_adv.grad.zero_()

    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the
    score of target_y under a pretrained model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor (1, 3, H, W)
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    # Replace "pass" statement with your code
    
    output_score = model(img) # (1, 1000)
    sc_I = output_score[:, target_y].mean() # scalar

    loss = sc_I - l2_reg * (torch.norm(img, p=2)**2) # MAXIMIZE
    
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
      img.data += learning_rate * img.grad # Gradient Ascent

    img.grad.zero_() # make gradients to ZERO

    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
