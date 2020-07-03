
import torch                                                                                          
                                                                                                      
def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)                                             
                                                                                                      
                                            
    
if __name__ == '__main__': 
    def f(x):                                                                                             
        return x * x * torch.arange(4, dtype=torch.float)                                                                                                        
    x = torch.ones(4, requires_grad=True)                                                                 
    print(jacobian(f(x), x))                                                                              
    print(hessian(f(x), x))  