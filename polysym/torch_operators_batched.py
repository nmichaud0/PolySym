import torch
from typing import Union, Callable, List, Dict

# Broadcasting wrapper functions for binary operations
def batched_binary_op(op_func):
    """
    Wrapper for binary operations that ensures proper broadcasting between
    scalar batches [n_observations] and vector batches [n_observations, timesteps].
    
    Args:
        op_func: Binary operation function to wrap (e.g., torch.add)
        
    Returns:
        Wrapped function that handles the broadcasting
    """
    def wrapper(a, b):
        # Check dimensions of inputs
        a_dim = a.dim()
        b_dim = b.dim()
        
        # If both have the same dimension, just apply the operation directly
        if a_dim == b_dim:
            return op_func(a, b)
        
        # Handle case where a is [n_observations] and b is [n_observations, timesteps]
        if a_dim == 1 and b_dim == 2 and a.size(0) == b.size(0):
            # Reshape a to [n_observations, 1] to broadcast across timesteps
            a_reshaped = a.view(-1, 1)
            return op_func(a_reshaped, b)
        
        # Handle case where b is [n_observations] and a is [n_observations, timesteps]
        if b_dim == 1 and a_dim == 2 and b.size(0) == a.size(0):
            # Reshape b to [n_observations, 1] to broadcast across timesteps
            b_reshaped = b.view(-1, 1)
            return op_func(a, b_reshaped)
        
        # If dimensions don't match in a compatible way, try PyTorch's default broadcasting
        # This will raise an error if broadcasting is not possible
        return op_func(a, b)
    
    return wrapper

# Safe versions of operations that can cause numerical issues
# These operations handle batched inputs (preserving observation dimension)

def safe_sqrt(x):
    return torch.sqrt(torch.abs(x) + 1e-10)

def safe_log(x):
    return torch.log(torch.abs(x) + 1e-10)

def safe_log10(x):
    return torch.log10(torch.abs(x) + 1e-10)

def safe_inv(x):
    return 1.0 / (x + torch.sign(x) * 1e-10)  # Add small value with same sign

def safe_div(a, b):
    """Safe division with proper broadcasting for batched tensors"""
    # Add small value with same sign to denominator to avoid division by zero
    return batched_binary_op(lambda x, y: x / (y + torch.sign(y) * 1e-10))(a, b)

def safe_pow(a, b):
    """Safe power with proper broadcasting for batched tensors"""
    # Avoid negative bases with non-integer exponents
    return batched_binary_op(lambda x, y: torch.pow(torch.abs(x) + 1e-10, y))(a, b)

def safe_asin(x):
    return torch.arcsin(torch.clamp(x, -1.0 + 1e-10, 1.0 - 1e-10))

def safe_acos(x):
    return torch.arccos(torch.clamp(x, -1.0 + 1e-10, 1.0 - 1e-10))

# Reduction operators that preserve batch dimension
def batched_mean(x):
    """Compute mean along last dimension for 2D tensors, preserving batch dimension"""
    if x.dim() > 1:
        return torch.mean(x, dim=-1)
    return x  # Already a 1D tensor of scalars

def batched_median(x):
    """Compute median along last dimension for 2D tensors, preserving batch dimension"""
    if x.dim() > 1:
        return torch.median(x, dim=-1)[0]
    return x  # Already a 1D tensor of scalars

def batched_std(x):
    """Compute standard deviation along last dimension for 2D tensors, preserving batch dimension"""
    if x.dim() > 1:
        return torch.std(x, dim=-1, unbiased=False)
    return x  # Already a 1D tensor of scalars

def batched_sum(x):
    """Compute sum along last dimension for 2D tensors, preserving batch dimension"""
    if x.dim() > 1:
        return torch.sum(x, dim=-1)
    return x  # Already a 1D tensor of scalars

def batched_min(x):
    """Compute minimum along last dimension for 2D tensors, preserving batch dimension"""
    if x.dim() > 1:
        return torch.min(x, dim=-1)[0]
    return x  # Already a 1D tensor of scalars

def batched_max(x):
    """Compute maximum along last dimension for 2D tensors, preserving batch dimension"""
    if x.dim() > 1:
        return torch.max(x, dim=-1)[0]
    return x  # Already a 1D tensor of scalars

# Simple element-wise operators - these preserve dimensions naturally
_unary_operators_map = {
    'neg': torch.neg,
    'abs': torch.abs,
    'sqrt': safe_sqrt,
    'exp': torch.exp,
    'log': safe_log,
    'log10': safe_log10,
    'inv': safe_inv,

    'sin': torch.sin,
    'cos': torch.cos,
    'tan': torch.tan,
    'asin': safe_asin,
    'acos': safe_acos,
    'atan': torch.arctan,

    'sinh': torch.sinh,
    'cosh': torch.cosh,
    'tanh': torch.tanh,

    # Reduction operators that maintain batch dimension
    'mean': batched_mean,
    'median': batched_median,
    'std': batched_std,
    'sum': batched_sum,
    'min': batched_min,
    'max': batched_max,

    'sign': torch.sign,
}

# Binary operators with proper broadcasting behavior
_binary_operators_map = {
    'add': batched_binary_op(torch.add),
    'sub': batched_binary_op(torch.sub),
    'mul': batched_binary_op(torch.mul),
    'div': safe_div,  # Already using batched_binary_op internally
    'pow': safe_pow,  # Already using batched_binary_op internally
}


class BatchedOperators:
    def __init__(self, operators_selection: list[str]=None, select_all: bool=False):
        """
        Initialize operators for batched tensor operations.
        
        Args:
            operators_selection: List of operator names to use. If None, use all operators if select_all is True.
            select_all: If True and operators_selection is None, select all available operators.
        """
        self.operators_selection = operators_selection

        if self.operators_selection is None:
            if select_all is False:
                raise ValueError("operators_selection must be provided unless select_all is True")
            
            else:
                self.unary_operators = _unary_operators_map.copy()
                self.binary_operators = _binary_operators_map.copy()

        else:
            self.unary_operators = {k: v for k, v in _unary_operators_map.items() if k in operators_selection}
            self.binary_operators = {k: v for k, v in _binary_operators_map.items() if k in operators_selection}
        
    def get_unary(self) -> List[Callable]:
        """Get list of unary operator functions"""
        return list(self.unary_operators.values())
    
    def get_binary(self) -> List[Callable]:
        """Get list of binary operator functions"""
        return list(self.binary_operators.values())
    
    def get_unary_dict(self) -> Dict[str, Callable]:
        """Get dictionary of unary operator functions with names"""
        return self.unary_operators
    
    def get_binary_dict(self) -> Dict[str, Callable]:
        """Get dictionary of binary operator functions with names"""
        return self.binary_operators