import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from torch_scatter import scatter_mean
import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import torch_geometric

from .registry import register_model
from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2,EquivariantLayerNormV3
from .fast_layer_norm import EquivariantLayerNormFast
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .gaussian_rbf import GaussianRadialBasisLayer


def trace_grad_fn(tensor, visited=None):
    if visited is None:
        visited = set()

    # 如果当前张量是叶子节点，直接返回
    if tensor.grad_fn is None:
        print(f"Reached a leaf tensor: {tensor.shape}")
        return

    # 如果当前 grad_fn 已经被访问过，检测到循环
    if tensor.grad_fn in visited:
        print(f"Detected a cycle in the computation graph at grad_fn: {tensor.grad_fn}")
        return

    # 打印当前张量和 grad_fn
    print(f"Current tensor: {tensor.shape}, grad_fn: {tensor.grad_fn}")
    visited.add(tensor.grad_fn)

    # 递归遍历 next_functions
    for next_fn, _ in tensor.grad_fn.next_functions:
        if next_fn is not None:
            # 尝试获取 next_fn 的输入张量
            try:
                # 获取 next_fn 的输入张量
                input_tensors = next_fn.next_inputs()
                for input_tensor in input_tensors:
                    trace_grad_fn(input_tensor, visited)
            except AttributeError:
                # 如果 next_fn 没有 next_inputs 方法，跳过
                continue


torch.set_printoptions(profile='full')

_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 86
_AVG_NUM_NODES = 29.891087392943284
_AVG_DEGREE = 34.29242574467496 

def get_norm_layer(norm_type):
    if norm_type == 'graph':
        return EquivariantGraphNorm
    elif norm_type == 'instance':
        return EquivariantInstanceNorm
    elif norm_type == 'layer':
        return EquivariantLayerNormV2
    elif norm_type == 'fast_layer':
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError('Norm type {} not supported.'.format(norm_type))

class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope
        
    
    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2
    
    
    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)
            

def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


class FullyConnectedTensorProductRescaleNorm(FullyConnectedTensorProductRescale):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None, norm_layer='graph'):
        
        super().__init__(irreps_in1, irreps_in2, irreps_out,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.norm = get_norm_layer(norm_layer)(self.irreps_out)
        
        
    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        return out
        

class FullyConnectedTensorProductRescaleNormSwishGate(FullyConnectedTensorProductRescaleNorm):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None, norm_layer='graph'):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization, norm_layer=norm_layer)
        self.gate = gate
        
        
    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        out = self.gate(out)
        return out
    

class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.gate = gate
        
        
    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out

def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output, 
    internal_weights=False, bias=True):
    '''
        The irreps of output is pre-determined. 
        `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []
    
    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))
        
    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output) #irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
            irreps_output, instructions,
            internal_weights=internal_weights,
            shared_weights=internal_weights,
            bias=bias, rescale=_RESCALE)
    return tp  

class SeparableFCTP(torch.nn.Module):
    '''
        Use separable FCTP for spatial convolution.
    '''
    def __init__(self, irreps_node_input, irreps_edge_attr, irreps_node_output, 
        fc_neurons, use_activation=False, norm_layer='graph', 
        internal_weights=False):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        norm = get_norm_layer(norm_layer)
        
        self.dtp = DepthwiseTensorProduct(self.irreps_node_input, self.irreps_edge_attr, 
            self.irreps_node_output, bias=False, internal_weights=internal_weights)
        
        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for (slice, slice_sqrt_k) in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k
                
        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)
        
        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)
        
        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated  # gated tensors
                )
            self.gate = gate
    
    
    def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs):
        '''
            Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by 
            self.dtp_rad(`edge_scalars`).
        '''
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:    
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out
        

@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
        Reshape vectors of shape [N, irreps_mid] to vectors of shape
        [N, num_heads, irreps_head].
    '''
    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(
            self.__class__.__name__, self.irreps_head, self.num_heads)
    
    
@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
        Convert vectors of shape [N, num_heads, irreps_head] into
        vectors of shape [N, irreps_head * num_heads].
    '''
    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)


class ConcatIrrepsTensor(torch.nn.Module):
    
    def __init__(self, irreps_1, irreps_2):
        super().__init__()
        assert irreps_1 == irreps_1.simplify()
        self.check_sorted(irreps_1)
        assert irreps_2 == irreps_2.simplify()
        self.check_sorted(irreps_2)
        
        self.irreps_1 = irreps_1
        self.irreps_2 = irreps_2
        self.irreps_out = irreps_1 + irreps_2
        self.irreps_out, _, _ = sort_irreps_even_first(self.irreps_out) #self.irreps_out.sort()
        self.irreps_out = self.irreps_out.simplify()
        
        self.ir_mul_list = []
        lmax = max(irreps_1.lmax, irreps_2.lmax)
        irreps_max = []
        for i in range(lmax + 1):
            irreps_max.append((1, (i, -1)))
            irreps_max.append((1, (i,  1)))
        irreps_max = o3.Irreps(irreps_max)
        
        start_idx_1, start_idx_2 = 0, 0
        dim_1_list, dim_2_list = self.get_irreps_dim(irreps_1), self.get_irreps_dim(irreps_2)
        for _, ir in irreps_max:
            dim_1, dim_2 = None, None
            index_1 = self.get_ir_index(ir, irreps_1)
            index_2 = self.get_ir_index(ir, irreps_2)
            if index_1 != -1:
                dim_1 = dim_1_list[index_1]
            if index_2 != -1:
                dim_2 = dim_2_list[index_2]
            self.ir_mul_list.append((start_idx_1, dim_1, start_idx_2, dim_2))
            start_idx_1 = start_idx_1 + dim_1 if dim_1 is not None else start_idx_1
            start_idx_2 = start_idx_2 + dim_2 if dim_2 is not None else start_idx_2
          
            
    def get_irreps_dim(self, irreps):
        muls = []
        for mul, ir in irreps:
            muls.append(mul * ir.dim)
        return muls
    
    
    def check_sorted(self, irreps):
        lmax = None
        p = None
        for _, ir in irreps:
            if p is None and lmax is None:
                p = ir.p
                lmax = ir.l
                continue
            if ir.l == lmax:
                assert p < ir.p, 'Parity order error: {}'.format(irreps)
            assert lmax <= ir.l                
        
    
    def get_ir_index(self, ir, irreps):
        for index, (_, irrep) in enumerate(irreps):
            if irrep == ir:
                return index
        return -1
    
    
    def forward(self, feature_1, feature_2):
        
        output = []
        for i in range(len(self.ir_mul_list)):
            start_idx_1, mul_1, start_idx_2, mul_2 = self.ir_mul_list[i]
            if mul_1 is not None:
                output.append(feature_1.narrow(-1, start_idx_1, mul_1))
            if mul_2 is not None:
                output.append(feature_2.narrow(-1, start_idx_2, mul_2))
        output = torch.cat(output, dim=-1)
        return output
    
    
    def __repr__(self):
        return '{}(irreps_1={}, irreps_2={})'.format(self.__class__.__name__, 
            self.irreps_1, self.irreps_2)

@compile_mode('script')
class GraphAttention(torch.nn.Module):
    '''
        1. Message = Alpha * Value
        2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
        3. 0e -> Activation -> Inner Product -> (Alpha)
        4. (0e+1e+...) -> (Value)
    '''
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_edge_attr, irreps_node_output,
        fc_neurons,
        irreps_head, num_heads, irreps_pre_attn=None, 
        rescale_degree=False, nonlinear_message=False,
        alpha_drop=0.1, proj_drop=0.1):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        
        # Merge src and dst
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)
        
        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads) #irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify() 
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps('{}x0e'.format(mul_alpha)) # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()
        
        self.sep_act = None
        if self.nonlinear_message:
            # Use an extra separable FCTP and Swish Gate for value
            self.sep_act = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, self.irreps_pre_attn, fc_neurons, 
                use_activation=True, norm_layer=None, internal_weights=False)
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_heads, fc_neurons=None, 
                use_activation=False, norm_layer=None, internal_weights=True)
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
                num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_all, fc_neurons, 
                use_activation=False, norm_layer=None)
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps('{}x0e'.format(mul_alpha_head)) + irreps_head).simplify(), 
                num_heads)
        
        self.alpha_act = Activation(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
            [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head)
        
        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        
        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_input, 
                drop_prob=proj_drop)
        
        
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, 
        batch, **kwargs):
        
        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]
        
        if self.nonlinear_message:          
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars)
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))
        
        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum('bik, aik -> bi', alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn = self.heads2vec(attn)
        
        if self.rescale_degree:
            degree = torch_geometric.utils.degree(edge_dst, 
                num_nodes=node_input.shape[0], dtype=node_input.dtype)
            degree = degree.view(-1, 1)
            attn = attn * degree
            
        node_output = self.proj(attn)
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str
                    

@compile_mode('script')
class FeedForwardNetwork(torch.nn.Module):
    '''
        Use two (FCTP + Gate)
    '''
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_node_output, irreps_mlp_mid=None,
        proj_drop=0.1):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        
        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_mlp_mid, 
            bias=True, rescale=_RESCALE)
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid, self.irreps_node_attr, self.irreps_node_output, 
            bias=True, rescale=_RESCALE)
        
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_output, 
                drop_prob=proj_drop)
            
        
    def forward(self, node_input, node_attr, **kwargs):
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output

@compile_mode('script')
class TransBlock(torch.nn.Module):
    '''
        1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
        2. Use pre-norm architecture
    '''
    
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_edge_attr, irreps_node_output,
        fc_neurons,
        irreps_head, num_heads, irreps_pre_attn=None, 
        rescale_degree=False, nonlinear_message=False,
        alpha_drop=0.1, proj_drop=0.1,
        drop_path_rate=0.0,
        irreps_mlp_mid=None,
        ):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        
        self.resweight = torch.nn.Parameter(torch.Tensor([0]))
        self.ga = GraphAttention(irreps_node_input=self.irreps_node_input, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr, 
            irreps_node_output=self.irreps_node_input,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head, 
            num_heads=self.num_heads, 
            irreps_pre_attn=self.irreps_pre_attn, 
            rescale_degree=self.rescale_degree, 
            nonlinear_message=self.nonlinear_message,
            alpha_drop=alpha_drop, 
            proj_drop=proj_drop)
        
        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None
        
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input, #self.concat_norm_output.irreps_out, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output, 
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop)
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input, self.irreps_node_attr, 
                self.irreps_node_output, 
                bias=True, rescale=_RESCALE)
            
            
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, 
        batch, **kwargs):
        
        node_output = node_input
        node_features = node_input
        node_features = self.ga(node_input=node_features, 
            node_attr=node_attr, 
            edge_src=edge_src, edge_dst=edge_dst, 
            edge_attr=edge_attr, edge_scalars=edge_scalars,
            batch=batch)
        node_features = node_features*self.resweight
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features
        
        node_features = node_output
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)
        
        node_features = node_features*self.resweight
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features
        
        return node_output
    

class NodeEmbeddingNetwork(torch.nn.Module):
    
    def __init__(self, irreps_node_embedding, max_atom_type=_MAX_ATOM_TYPE, bias=True):
        
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(o3.Irreps('{}x0e'.format(self.max_atom_type)), 
            self.irreps_node_embedding, bias=bias)
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type ** 0.5)
        
        
    def forward(self, node_atom):
        '''
            `node_atom` is a LongTensor.
        '''
        node_atom_onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).float()
        node_attr = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)
        
        return node_embedding, node_attr, node_atom_onehot

class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0


    def forward(self, x, index, **kwargs):
        out = scatter(x, index, reduce='mean',**kwargs)
        out = out.div(self.avg_aggregate_num ** 0.5)
        return out
    
    
    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.avg_aggregate_num)

class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num):
        super().__init__()
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding, 
            bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding, 
            irreps_edge_attr, irreps_node_embedding, 
            internal_weights=False, bias=False)
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for (slice, slice_sqrt_k) in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)
        
    
    def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch):
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)

        # print(node_features[edge_src].shape,edge_attr.shape)

        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(edge_features, edge_dst, dim=0, 
            dim_size=node_features.shape[0])
        return node_features
    
class GraphAttentionTransformer(torch.nn.Module):
    def __init__(self,
        irreps_in='86x0e',
        irreps_out='1x0e+1x1o+1x2e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0, edge_num = 0,
        mean=None, std=None, scale=None, atomref=None):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.edge_num = edge_num
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE), 
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps(irreps_out), rescale=_RESCALE)) 
        self.lrs = LinearRS(o3.Irreps('86x0e'), o3.Irreps('512x0e'), rescale=_RESCALE)
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        self.atom_expand = LinearRS(o3.Irreps('86x0e'), o3.Irreps('128x0e+64x1e+32x2e'), rescale=_RESCALE)
        self.apply(self._init_weights)
        
        
    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads, 
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                )
            self.blocks.append(blk)
            
            
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
                          
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer)): 
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
                    
        return set(no_wd_list)
        

    def forward(self, f_in, edge_src,edge_dst, edge_vec, edge_attr, edge_num,
                batch, **kwargs) -> torch.Tensor:
        
        f_in = f_in.to(torch.float32) #x
        batch_counts = torch.cat((
            torch.zeros(1,dtype=torch.long,device=f_in.device),
            torch.cumsum(torch.bincount(batch),dim=0)[:-1]
        ))

        counts = torch.cat([batch_counts[i].repeat(edge_num[i])  for i in range(len(edge_num))])
        edge_src.add_(counts)
        edge_dst.add_(counts)
       
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
        
        atom_embedding = f_in
        atom_embedding = self.atom_expand(atom_embedding)
        edge_length = edge_attr 
        # edge_occu = edge_occu.unsqueeze(-1)
        # edge_occu = edge_occu.expand(-1, self.number_of_basis)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src-1, edge_dst-1, batch)
        node_features = atom_embedding + edge_degree_embedding #480

        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        
        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src-1, edge_dst=edge_dst-1, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch)
        
        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
            
        edge_embedding = self.lrs(edge_length_embedding)
        node_features = node_features[edge_src-1] + node_features[edge_dst-1] + edge_embedding #512
        outputs = self.head(node_features)
        # outputs = self.scale_scatter(outputs, batch, dim=0)
        
        # if self.scale is not None:
        #     outputs = self.scale * outputs
        return outputs

class GraphAttentionTransformer_dx(torch.nn.Module):
    def __init__(self,
        irreps_in='86x0e',
        irreps_out='1x0e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0, edge_num = 0,
        mean=None, std=None, scale=None, atomref=None):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.edge_num = edge_num
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE), 
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps(irreps_out), rescale=_RESCALE)) 
        self.lrs = LinearRS(o3.Irreps('86x0e'), o3.Irreps('512x0e'), rescale=_RESCALE)
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        self.atom_expand = LinearRS(o3.Irreps('86x0e'), o3.Irreps('128x0e+64x1e+32x2e'), rescale=_RESCALE)
        self.apply(self._init_weights)
        
        
    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads, 
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                )
            self.blocks.append(blk)
            
            
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
                          
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer)): 
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
                    
        return set(no_wd_list)
        

    def forward(self, f_in, edge_src,edge_dst, pos, edge_num,
                batch, device, **kwargs) -> torch.Tensor:
        diff = pos[:, None, :] - pos[None, :, :]
        dist_squared = torch.sum(diff**2, dim=-1)
        dist_squared += (torch.eye(dist_squared.shape[0]) * 1e-10).to(device)
        dist_matrix = torch.sqrt(dist_squared)
        threshold = 8
        max_neighbors=32
        edge_vec_list, distances_list = [], []
        num_atoms_per_sample = torch.bincount(batch).tolist()
        start_index = 0
        for num_atoms in num_atoms_per_sample:
            end_index = start_index + num_atoms
            sample_pos = pos[start_index:end_index]  # 当前样本的原子位置
            diff = sample_pos[:, None, :] - sample_pos[None, :, :]
            dist_squared = torch.sum(diff**2, dim=-1)
            dist_squared += (torch.eye(dist_squared.shape[0]) * 1e-10).to(device)
            dist_matrix = torch.sqrt(dist_squared)
            edge_vec, distances = [], []
            for i in range(num_atoms):
                neighbors_indices = dist_matrix[i].argsort()[1:max_neighbors+1]
                for neighbor_index in neighbors_indices:
                    if dist_matrix[i, neighbor_index] > threshold:
                        continue
                    if neighbor_index == i:
                        continue
                    edge_vector = sample_pos[i] - sample_pos[neighbor_index]
                    edge_vec.append(edge_vector)
                    distances.append(dist_matrix[i, neighbor_index])
            edge_vec_list.append(torch.stack(edge_vec, dim=0))
            distances_list.append(torch.tensor(distances, device=device))
            start_index = end_index
        edge_vec = torch.cat(edge_vec_list, dim=0)
        edge_attr = torch.cat(distances_list, dim=0)

        f_in = f_in.to(torch.float32) #x
        batch_counts = torch.cat((
            torch.zeros(1,dtype=torch.long,device=f_in.device),
            torch.cumsum(torch.bincount(batch),dim=0)[:-1]
        ))

        counts = torch.cat([batch_counts[i].repeat(edge_num[i])  for i in range(len(edge_num))])
        edge_src.add_(counts)
        edge_dst.add_(counts)

        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')

        atom_embedding = f_in
        atom_embedding = self.atom_expand(atom_embedding)

        edge_length = edge_attr 
        # edge_occu = edge_occu.unsqueeze(-1)
        # edge_occu = edge_occu.expand(-1, self.number_of_basis)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src-1, edge_dst-1, batch)
        node_features = atom_embedding + edge_degree_embedding #480

        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        
        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src-1, edge_dst=edge_dst-1, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch)
        
        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
            
        edge_embedding = self.lrs(edge_length_embedding)
        node_features = node_features[edge_src-1] + node_features[edge_dst-1] + edge_embedding #512
        outputs = self.head(node_features)
        # outputs = self.scale_scatter(outputs, batch, dim=0)
        
        if self.scale is not None:
            outputs = self.scale * outputs
        edge_batch = batch[edge_src-1]
        return scatter_mean(outputs, edge_batch, dim=0)

class GraphAttentionTransformer_dx_v2(torch.nn.Module):
    def __init__(self,
        irreps_in='86x0e',
        irreps_out='1x0e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0, edge_num = 0,
        mean=None, std=None, scale=None, atomref=None):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.edge_num = edge_num
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE), 
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps(irreps_out), rescale=_RESCALE)) 
        self.lrs = LinearRS(o3.Irreps('86x0e'), o3.Irreps('512x0e'), rescale=_RESCALE)
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        self.atom_expand = LinearRS(o3.Irreps('86x0e'), o3.Irreps('128x0e+64x1e+32x2e'), rescale=_RESCALE)
        self.apply(self._init_weights)
        
        
    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads, 
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                )
            self.blocks.append(blk)
            
            
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
                          
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer)): 
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
                    
        return set(no_wd_list)
        

    def forward(self, f_in, edge_src,edge_dst, pos, edge_num,
                batch, device, **kwargs) -> torch.Tensor:
        diff = pos[:, None, :] - pos[None, :, :]
        dist_squared = torch.sum(diff**2, dim=-1)
        dist_squared += (torch.eye(dist_squared.shape[0]) * 1e-10).to(device)
        dist_matrix = torch.sqrt(dist_squared)
        threshold = 8
        max_neighbors=32
        edge_vec, distances = [], []
        num_atoms = pos.shape[0]
        for i in range(num_atoms):
            # 获取当前原子与其他所有原子的距离并排序获取索引
            # 由于argsort是升序，排除了第一个索引（即自身），接着取前k个邻居
            neighbors_indices = dist_matrix[i].argsort()[1:max_neighbors+1]
            for neighbor_index in neighbors_indices:
                if dist_matrix[i, neighbor_index] > threshold:
                    continue
                # 确保不是自连接
                if neighbor_index == i:
                    continue
                # 计算并添加边向量
                edge_vector = pos[i] - pos[neighbor_index]
                edge_vec.append(edge_vector)
                # 添加边的距离
                distances.append(dist_matrix[i, neighbor_index])
        edge_vec = torch.stack(edge_vec, dim=0)  # 堆叠列表中的向量
        edge_attr = torch.as_tensor(distances, dtype=torch.float, device=device)

        f_in = f_in.to(torch.float32) #x
        batch_size = batch.max().item()+1
        edge_src_batched = edge_src + torch.arange(batch_size , device = f_in.device).repeat_interleave(edge_num)
        edge_dst_batched = edge_dst + torch.arange(batch_size , device = f_in.device).repeat_interleave(edge_num)

        batch_counts = torch.cat((
            torch.zeros(1,dtype=torch.long,device=f_in.device),
            torch.cumsum(torch.bincount(batch),dim=0)[:-1]
        ))

        counts = torch.cat([batch_counts[i].repeat(edge_num[i])  for i in range(len(edge_num))])
        edge_src.add_(counts)
        edge_dst.add_(counts)
       
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
        
        atom_embedding = f_in
        atom_embedding = self.atom_expand(atom_embedding)
        edge_length = edge_attr 
        # edge_occu = edge_occu.unsqueeze(-1)
        # edge_occu = edge_occu.expand(-1, self.number_of_basis)
        edge_length_embedding = self.rbf(edge_length)
        # edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
        # edge_length_embedding, edge_src-1, edge_dst-1, batch)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
        edge_length_embedding, edge_src_batched, edge_dst_batched, batch)
        node_features = atom_embedding + edge_degree_embedding #480

        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        
        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src-1, edge_dst=edge_dst-1, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch)
        
        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
            
        edge_embedding = self.lrs(edge_length_embedding)
        node_features = node_features[edge_src-1] + node_features[edge_dst-1] + edge_embedding #512
        outputs = self.head(node_features)
        # outputs = self.scale_scatter(outputs, batch, dim=0)
        
        if self.scale is not None:
            outputs = self.scale * outputs
        return torch.mean(outputs)

class GraphAttentionTransformer_dx_v3(torch.nn.Module):
    def __init__(self,
                 irreps_in='86x0e',
                 irreps_out='1x0e',
                 irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                 irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                 max_radius=5.0, number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64],
                 irreps_feature='512x0e',
                 irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
                 rescale_degree=False, nonlinear_message=False,
                 irreps_mlp_mid='128x0e+64x1e+32x2e',
                 norm_layer='layer',
                 alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
                 drop_path_rate=0.0, edge_num=0,
                 mean=None, std=None, scale=None, atomref=None):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.edge_num = edge_num
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.irreps_node_embedding.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        # Layers
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop) if self.out_drop != 0.0 else None
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE), 
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps(irreps_out), rescale=_RESCALE)
        ) 
        self.lrs = LinearRS(o3.Irreps('86x0e'), o3.Irreps('512x0e'), rescale=_RESCALE)
        self.atom_expand = LinearRS(o3.Irreps('86x0e'), self.irreps_node_embedding, rescale=_RESCALE)
        self.apply(self._init_weights)
    
    def build_blocks(self):
        for i in range(self.num_layers):
            irreps_block_output = self.irreps_node_embedding if i != (self.num_layers - 1) else self.irreps_feature
            blk = TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons,
                irreps_head=self.irreps_head,
                num_heads=self.num_heads,
                irreps_pre_attn=self.irreps_pre_attn,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                irreps_mlp_mid=self.irreps_mlp_mid,
            )
            self.blocks.append(blk)
            
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, f_in, edge_src, edge_dst, pos, edge_num, batch, device, **kwargs) -> torch.Tensor:
        # print(f"Initial data.pos.requires_grad: {pos.requires_grad}")
        # trace_grad_fn(pos)

        # 计算距离矩阵
        diff = pos[:, None, :] - pos[None, :, :]
        # print("看看diff的gradfn")
        # trace_grad_fn(diff)

        dist_matrix = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-10).to(device)
        # print("看看dist_matrix的gradfn")
        # trace_grad_fn(dist_matrix)

        threshold = 8
        max_neighbors = 32
        edge_vec_list, distances_list = [], []
        num_atoms_per_sample = torch.bincount(batch).tolist()
        start_index = 0

        # 获取当前样本的原子数
        num_atoms = dist_matrix.size(0)
        # 如果原子数目小于 max_neighbors + 1，则减少 k 的值
        k_value = min(max_neighbors + 1, num_atoms)

        # 计算每个原子最近的 max_neighbors 个原子
        _, nearest_indices = dist_matrix.topk(k=k_value, largest=False, sorted=False)
        nearest_distances = dist_matrix.gather(1, nearest_indices[:, 1:])  # 去掉对角线

        # 计算每个原子最大邻居距离，并与阈值取最小值
        max_neighbor_dist = nearest_distances.max(dim=1).values
        mask_threshold = torch.minimum(max_neighbor_dist, torch.tensor(threshold, device=device))

        # 创建掩码
        mask = (dist_matrix < mask_threshold[:, None]) & (dist_matrix > 1e-10)
        diagonal_mask = torch.eye(dist_matrix.size(0), dtype=torch.bool, device=device)
        mask = mask & ~diagonal_mask

        # 提取边的索引
        edge_src, edge_dst = torch.where(mask)
        edge_vec = pos[edge_src] - pos[edge_dst]
        # print("看看edge_vec的gradfn")
        # trace_grad_fn(edge_vec)

        edge_attr = dist_matrix[edge_src, edge_dst]
        # print("看看edge_attr的gradfn")
        # trace_grad_fn(edge_attr)

        # 球谐函数
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr, x=edge_vec, normalize=True, normalization='component')
        # print("看看edge_sh的gradfn")
        # trace_grad_fn(edge_sh)

        edge_length_embedding = self.rbf(edge_attr)
        # print("看看edge_length_embedding的gradfn")
        # trace_grad_fn(edge_length_embedding)

        # 节点和边的嵌入
        f_in = f_in.to(torch.float32)
        atom_embedding = self.atom_expand(f_in)
        # print("看看atom_embedding的gradfn")
        # trace_grad_fn(atom_embedding)

        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, batch)
        # print("看看edge_degree_embedding的gradfn")
        # trace_grad_fn(edge_degree_embedding)

        # 消息传递
        node_features = atom_embedding + edge_degree_embedding
        # print("看看node_features的gradfn (before blocks)")
        # trace_grad_fn(node_features)

        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr, edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, edge_scalars=edge_length_embedding, batch=batch)
            # print("看看每个block之后的node_features的gradfn")
            # trace_grad_fn(node_features)

        # 输出
        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
        edge_embedding = self.lrs(edge_length_embedding)
        node_features = node_features[edge_src] + node_features[edge_dst] + edge_embedding
        # print("看看forward里面决定outputs的node_features的gradfn")
        # trace_grad_fn(node_features)

        outputs = self.head(node_features)
        # print("看看forward里面经过head之前的outputs的gradfn")
        # trace_grad_fn(outputs)

        if self.scale is not None:
            outputs = self.scale * outputs
        # print("看看forward里面outputs的gradfn")
        # trace_grad_fn(outputs)

        scatter_mean_output = scatter_mean(outputs, batch[edge_src], dim=0)
        # print("看看 scatter_mean 的输出和其 grad_fn")
        # print(f"scatter_mean 的输出: {scatter_mean_output.shape}, grad_fn: {scatter_mean_output.grad_fn}")
        # trace_grad_fn(scatter_mean_output)

        # 返回 scatter_mean 的结果
        return scatter_mean_output


class GraphAttentionTransformer_dx_v4(torch.nn.Module):
    def __init__(self,
                 irreps_in='86x0e',
                 irreps_out='1x0e',
                 irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                 irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                 max_radius=5.0, number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64],
                 irreps_feature='512x0e',
                 irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
                 rescale_degree=False, nonlinear_message=False,
                 irreps_mlp_mid='128x0e+64x1e+32x2e',
                 norm_layer='layer',
                 alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
                 drop_path_rate=0.0, edge_num=0,
                 mean=None, std=None, scale=None, atomref=None):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.edge_num = edge_num
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.irreps_node_embedding.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        # Layers
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop) if self.out_drop != 0.0 else None
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE), 
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps(irreps_out), rescale=_RESCALE)
        ) 
        self.lrs = LinearRS(o3.Irreps('86x0e'), o3.Irreps('512x0e'), rescale=_RESCALE)
        self.atom_expand = LinearRS(o3.Irreps('86x0e'), self.irreps_node_embedding, rescale=_RESCALE)
        self.apply(self._init_weights)
    
    def build_blocks(self):
        for i in range(self.num_layers):
            irreps_block_output = self.irreps_node_embedding if i != (self.num_layers - 1) else self.irreps_feature
            blk = TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons,
                irreps_head=self.irreps_head,
                num_heads=self.num_heads,
                irreps_pre_attn=self.irreps_pre_attn,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                irreps_mlp_mid=self.irreps_mlp_mid,
            )
            self.blocks.append(blk)
            
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, f_in, edge_src, edge_dst, pos, edge_num, batch, device, **kwargs) -> torch.Tensor:
        # 初始化一些列表，用于保存逐样本的计算结果
        edge_vec_list, distances_list, edge_src_list, edge_dst_list = [], [], [], []

        # 计算每个样本的原子数量
        num_atoms_per_sample = torch.bincount(batch).tolist()
        start_index = 0

        for num_atoms in num_atoms_per_sample:
            end_index = start_index + num_atoms

            # 当前样本的原子位置
            sample_pos = pos[start_index:end_index]  # Shape: (num_atoms, 3)

            # 计算样本内的距离矩阵
            diff = sample_pos[:, None, :] - sample_pos[None, :, :]
            dist_matrix = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-10)  # Shape: (num_atoms, num_atoms)

            # 限制邻居的数量和距离范围
            max_neighbors = 32
            threshold = 8
            _, nearest_indices = dist_matrix.topk(k=max_neighbors + 1, largest=False, sorted=False)
            nearest_distances = dist_matrix.gather(1, nearest_indices[:, 1:])  # 去掉自身的距离

            # 创建掩码，限制最大邻居距离
            max_neighbor_dist = nearest_distances.max(dim=1).values
            mask_threshold = torch.minimum(max_neighbor_dist, torch.tensor(threshold, device=device))
            mask = (dist_matrix < mask_threshold[:, None]) & (dist_matrix > 1e-10)

            # 提取边索引和属性
            edge_src_sample, edge_dst_sample = mask.nonzero(as_tuple=True)
            edge_vec_sample = sample_pos[edge_src_sample] - sample_pos[edge_dst_sample]
            edge_attr_sample = dist_matrix[edge_src_sample, edge_dst_sample]

            # 保存样本内的边特性和索引
            edge_vec_list.append(edge_vec_sample)
            distances_list.append(edge_attr_sample)
            edge_src_list.append(edge_src_sample + start_index)
            edge_dst_list.append(edge_dst_sample + start_index)

            start_index = end_index

        # 将每个样本的边信息拼接到全局张量中
        edge_vec = torch.cat(edge_vec_list, dim=0)  # Shape: (total_edges, 3)
        edge_attr = torch.cat(distances_list, dim=0)  # Shape: (total_edges,)
        edge_src = torch.cat(edge_src_list, dim=0)  # Shape: (total_edges,)
        edge_dst = torch.cat(edge_dst_list, dim=0)  # Shape: (total_edges,)

        # 球谐函数
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr, x=edge_vec, normalize=True, normalization='component'
        )

        # 节点嵌入
        f_in = f_in.to(torch.float32)
        atom_embedding = self.atom_expand(f_in)

        # 边嵌入
        edge_length_embedding = self.rbf(edge_attr)
        edge_degree_embedding = self.edge_deg_embed(
            atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, batch
        )

        # 消息传递
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        for blk in self.blocks:
            node_features = blk(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch
            )

        # 节点归一化和输出
        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)

        edge_embedding = self.lrs(edge_length_embedding)
        node_features = node_features[edge_src] + node_features[edge_dst] + edge_embedding
        outputs = self.head(node_features)

        if self.scale is not None:
            outputs = self.scale * outputs

        # 聚合样本级结果
        edge_batch = batch[edge_src]
        return scatter_mean(outputs, edge_batch, dim=0)



@register_model
def graph_attention_transformer_l2_noNorm(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='16x0e+8x1e+4x2e', num_heads=8, irreps_pre_attn=None,
	    rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_l2_noNorm(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='16x0e+8x1e+4x2e', num_heads=8, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_l2_e3_noNorm(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_out='1x0e+1x1o+1x2e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
	    irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='16x0e+8x1e+4x2e', num_heads=8, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model

@register_model
def graph_attention_transformer_nonlinear_l2_e3_noNorm_dx(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer_dx_v3(
        irreps_in=irreps_in,
        irreps_out='1x0e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
	    irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='16x0e+8x1e+4x2e', num_heads=8, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model
