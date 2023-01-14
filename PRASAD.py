import torch
import numpy as np
import torch.nn as nn

cuda = torch.cuda.is_available()
device = torch.device("cpu")

def DataLoader_train(train, batch_size=16):
    batch_idx = np.random.choice(len(train.targets), size=batch_size)
    X_train = train.data[batch_idx].float().reshape(batch_size, -1)
    y_train = train.targets[batch_idx].long()
        
    return X_train, y_train

def shift_test(w):
    b = w < 0
    s = np.log2(np.abs(w)).astype("int")
    return b, s

def shift_inference(b, s):
    w_one = np.float_power(2, s)
    w_one[np.where(b)] = -w_one[np.where(b)]
    return w_one

def torch_train(net, criterion, lr, train, number_of_batches):
    for each_batch in range(number_of_batches):
        data, target = DataLoader_train(train)   
        data = data.to(device)
        target = target.to(device)
        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, target)    
        loss.backward()

        net0_weight = net[0].weight.detach().cpu().numpy()
        net2_weight = net[2].weight.detach().cpu().numpy()
        
        net0_grad = net[0].weight.grad.detach().cpu().numpy()
        net2_grad = net[2].weight.grad.detach().cpu().numpy()
        
        net0_weight -= lr * net0_grad
        net2_weight -= lr * net2_grad
        
        net[0].weight = nn.Parameter(torch.Tensor(net0_weight))
        net[2].weight = nn.Parameter(torch.Tensor(net2_weight))
        
    return net

def torch_eval(net, test):
    with torch.no_grad():
        net.eval()

        net0_weight = net[0].weight.detach().cpu().numpy()
        net2_weight = net[2].weight.detach().cpu().numpy()
        
        b0, s0 = shift_test(net0_weight); w_one0 = shift_inference(b0, s0)
        b2, s2 = shift_test(net2_weight); w_one2 = shift_inference(b2, s2)
    
        net[0].weight = nn.Parameter(torch.Tensor(w_one0))
        net[2].weight = nn.Parameter(torch.Tensor(w_one2))
        
        data = test.data.float().reshape(-1, 784).to(device)
        # targets = test.targets.long().to(device)
        targets = test.targets
        targets_shape0 = targets.shape[0]
        
        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)

    test_accuracy = int(sum(predicted == targets)) / targets_shape0
    print(test_accuracy)
    return test_accuracy # This return does not work for some reason
