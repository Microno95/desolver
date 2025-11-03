import torch
import torch.func
import desolver.integrators
import inspect
from desolver.differential_system import DiffRHS, OdeResult, solve_ivp
from torch.utils import _pytree as pytree


def torch_solve_ivp(fun, t_span, y0, method='RK45', events=None, vectorized=False, args=None, kwargs:dict|None=None, **options):
    fn = fun
    while isinstance(fn, DiffRHS):
        fn = fn.rhs
    fn_args_kwargs = inspect.getfullargspec(fn)
    system_parameters = {
        "fun": fun,
        "t_span": t_span,
        "y0": y0,
        "method": method,
        "events": events,
        "kwargs": kwargs if kwargs is not None else dict(),
        "options": options
    }
    if args is not None:
        system_parameters["kwargs"].update(dict(zip(fn_args_kwargs[0][2:], args)))
    parameter_keys = ["fun", "t_span", "y0", "method", "events", "kwargs", "options"]
    
    flattened_parameters, treespec = pytree.tree_flatten(system_parameters)
    tensor_parameters = [key for key,val in enumerate(flattened_parameters) if torch.is_tensor(val) and val.requires_grad]
    non_tensor_parameters = [key for key,val in enumerate(flattened_parameters) if not torch.is_tensor(val) or (torch.is_tensor(val) and not val.requires_grad)]
    
    class WrappedSolveIVP(torch.autograd.Function):
        @staticmethod
        def forward(*flattened_args):
            _system_parameters = pytree.tree_unflatten(flattened_args, treespec)
            options = _system_parameters.pop('options')
            options["dense_output"] = any(torch.is_tensor(i) and i.requires_grad for i in flattened_args)
            system_solution  = solve_ivp(**_system_parameters, **options)
            return system_solution.t.clone().contiguous(), system_solution.y.clone().contiguous(), system_solution


        @staticmethod
        def setup_context(ctx:torch.autograd.function.FunctionCtx, inputs:tuple[desolver.integrators.IntegratorTemplate,torch.Tensor], outputs:tuple[torch.Tensor]):
            tensors_to_save = [inputs[i] for i in tensor_parameters]
            objects_to_save = [inputs[i] for i in non_tensor_parameters]
            ctx.save_for_backward(outputs[0], outputs[1], *tensors_to_save)
            ctx.save_for_forward(outputs[0], outputs[1], *tensors_to_save)
            ctx.objects_to_save = objects_to_save
            ctx.atol, ctx.rtol = outputs[2]["ode_system"].integrator.atol, outputs[2]["ode_system"].integrator.rtol
            ctx.dense_sol = outputs[2]["ode_system"].sol
            
            
        @staticmethod
        def vjp(ctx:torch.autograd.function.FunctionCtx, temporal_cotangents:torch.Tensor, state_cotangents:torch.Tensor, *ignored_args):
            flattened_args = [None for _ in range(len(flattened_parameters))]
            evaluation_times, evaluation_states = ctx.saved_tensors[:2]
            for idx, jdx in enumerate(tensor_parameters):
                flattened_args[jdx] = ctx.saved_tensors[idx+2]
            for idx, jdx in enumerate(non_tensor_parameters):
                flattened_args[jdx] = ctx.objects_to_save[idx]
            
            _system_parameters = pytree.tree_unflatten(flattened_args, treespec)
            fun, t_span, _, method, _, kwargs, options = [_system_parameters[key] for key in parameter_keys]
            method = options.get("adjoint_method", method)
                
            input_grads = {key: None for key in parameter_keys}
            input_grads["events"] = None if events is None else [None]*len(events)
            input_grads["options"] = {key: None for key in options.keys()}
            if "callbacks" in input_grads["options"] and options["callbacks"] is not None:
                input_grads["options"]["callbacks"] = [None]*len(options["callbacks"])
            
            constants = dict()
            if kwargs is not None:
                constants.update(kwargs)
            
            tensor_constants = [key for key,val in constants.items() if torch.is_tensor(val) and val.requires_grad]
            non_tensor_constants = set(constants.keys()) - set(tensor_constants)
            non_tensor_constants = {key: constants[key] for key in non_tensor_constants}
            
            y_dim, y_shape = evaluation_states[...,-1].numel(), evaluation_states[...,-1].shape
            
            if len(tensor_constants) > 0:
                const_dims, const_shapes = [constants[key].numel() for key in tensor_constants], [constants[key].shape for key in tensor_constants]
                const_total_dim = sum(const_dims)
            else:
                const_total_dim = None
            
            def wrapped_rhs(t, y, kwargs):
                _constants = {key: value for key,value in zip(tensor_constants, kwargs)}
                return fun(t, y, **_constants, **non_tensor_constants)
            
            def augmented_reverse_fn(t, y, **kwargs):
                if const_total_dim is not None:
                    _y, _cot, _ = ctx.dense_sol(t).detach(), *torch.split(y, [y_dim, const_total_dim])
                else:
                    _y, _cot = ctx.dense_sol(t).detach(), *torch.split(y, [y_dim])
                _y, _cot = _y.view(y_shape), _cot.view(y_shape)
                _dydt, vjp = torch.func.vjp(wrapped_rhs, t, _y, [kwargs[key] for key in tensor_constants])
                _, _dcotdt, _dargs_dt = vjp(_cot, retain_graph=torch.is_grad_enabled())
                ret_dydt = torch.cat([
                    # _dydt.view(-1),
                    -torch.cat([
                        _dcotdt.view(-1),
                        *[i.view(-1) for i in _dargs_dt]
                    ], dim=-1)
                ], dim=-1)
                return ret_dydt
            
            cot_split = [(t,v.view(-1)) for t,v in zip(evaluation_times.unbind(-1), state_cotangents.unbind(-1)) if torch.any(v != 0.0)]
            cot_split = cot_split[::-1]
            if not torch.any(state_cotangents[...,0] != 0.0):
                cot_split = cot_split + [(evaluation_times[0], state_cotangents[...,0])]
            cot_tf, adj_tf = cot_split[0]
            
            augmented_y = torch.cat([
                # evaluation_states[...,nearest_state_index].view(-1),
                cot_split[0][1].view(-1),
            ], dim=-1)

            if len(tensor_constants) > 0:
                augmented_y = torch.cat([
                    augmented_y,
                    *[torch.zeros_like(constants[key].view(-1)) for key in tensor_constants]
                ], dim=-1)
            
            options["atol"], options["rtol"] = options.get("adjoint_atol", ctx.atol), options.get("adjoint_rtol", ctx.rtol)
            # options["atol"] = torch.cat([
            #     torch.ones_like(y0.view(-1))*options["atol"],
            #     torch.ones_like(augmented_y[y_dim:])*torch.inf
            # ], dim=-1)
            # options["rtol"] = torch.cat([
            #     torch.ones_like(y0.view(-1))*options["rtol"],
            #     torch.ones_like(augmented_y[y_dim:])*torch.inf
            # ], dim=-1)
            for cot_t0, cot_state in cot_split[1:]:
                res = torch_solve_ivp(augmented_reverse_fn, t_span=[cot_tf, cot_t0], y0=augmented_y, method=method, kwargs={key: constants[key] for key in tensor_constants}, **options)
                cot_tf = res.t[-1]
                augmented_y = res.y[...,-1] + torch.cat([
                    # torch.zeros_like(y0.view(-1)),
                    cot_state.view(-1),
                    *[torch.zeros_like(constants[key].view(-1)) for key in tensor_constants]
                ], dim=-1)
            
            if const_total_dim is not None:
                y_t0, adj_t0, args_tf = ctx.dense_sol(evaluation_times[0]), *torch.split(augmented_y, [y_dim, const_total_dim])
                args_tf = torch.split(args_tf, const_dims)
                args_tf = {key: v.view(s) for key,s,v in zip(tensor_constants, const_shapes, args_tf)}
            else:
                y_t0, adj_t0 = ctx.dense_sol(evaluation_times[0]), *torch.split(augmented_y, [y_dim])
                args_tf = None
            
            rhs_at_t0 = wrapped_rhs(
                cot_tf, y_t0.view(y_shape), [constants[key] for key in tensor_constants]
            )
            rhs_at_tf = wrapped_rhs(
                evaluation_times[...,-1], evaluation_states[...,-1].view(y_shape), [constants[key] for key in tensor_constants]
            )

            input_grads["t_span"] = (
                (temporal_cotangents[ 0] - torch.sum(adj_tf * rhs_at_tf.ravel())).view(temporal_cotangents[ 0].shape) if torch.is_tensor(t_span[0]) and t_span[0].requires_grad else None,
                (temporal_cotangents[-1] + torch.sum(adj_t0 * rhs_at_t0.ravel())).view(temporal_cotangents[-1].shape) if torch.is_tensor(t_span[1]) and t_span[1].requires_grad else None,                
            )
            
            input_grads["y0"] = adj_t0.view(y_shape)
            if kwargs is not None:
                input_grads["kwargs"] = {
                    key: args_tf[key] if key in tensor_constants else None for key in kwargs.keys()
                }
            
            return tuple(pytree.tree_flatten(input_grads)[0])
        
        
        @staticmethod
        def jvp(ctx:torch.autograd.function.FunctionCtx, *flattened_tangents):
            flattened_args = [None for _ in range(len(flattened_parameters))]
            evaluation_times, evaluation_states = ctx.saved_tensors[:2]
            for idx, jdx in enumerate(tensor_parameters):
                flattened_args[jdx] = ctx.saved_tensors[idx+2]
            for idx, jdx in enumerate(non_tensor_parameters):
                flattened_args[jdx] = ctx.objects_to_save[idx]
            
            _system_parameters = pytree.tree_unflatten(flattened_args, treespec)
            _system_tangents = pytree.tree_unflatten(flattened_tangents, treespec)
            fun, t_span, y0, method, events, kwargs, options = [_system_parameters[key] for key in parameter_keys]
            _, (t0_tangent, tf_tangent), y0_tangent, _, _, kwargs_tangents, _ = [_system_tangents[key] for key in parameter_keys]
                
            input_grads = {key: None for key in parameter_keys}
            input_grads["events"] = None if events is None else [None]*len(events)
            input_grads["options"] = {key: None for key in options.keys()}
            if "callbacks" in input_grads["options"] and options["callbacks"] is not None:
                input_grads["options"]["callbacks"] = [None]*len(options["callbacks"])
            
            constants = dict()
            if kwargs is not None:
                constants.update(kwargs)
            constants_tangents = dict()
            if kwargs_tangents is not None:
                constants_tangents.update(kwargs_tangents)
            
            tensor_constants = [key for key,val in constants_tangents.items() if torch.is_tensor(val)]
            non_tensor_constants = set(constants.keys()) - set(tensor_constants)
            non_tensor_constants = {key: constants[key] for key in non_tensor_constants}
            
            y_dim, y_shape = evaluation_states[...,-1].numel(), evaluation_states[...,-1].shape
            
            if len(tensor_constants) > 0:
                const_dims, const_shapes = [constants[key].numel() for key in tensor_constants], [constants[key].shape for key in tensor_constants]
                const_total_dim = sum(const_dims)
            else:
                const_total_dim = None
            
            def wrapped_rhs(t, y, *args):
                _constants = {key: value for key,value in zip(tensor_constants, args)}
                return fun(t, y, **_constants, **non_tensor_constants)
            
            def augmented_forward_fn(t, y, **kwargs):
                if const_total_dim is not None:
                    _y, _tan, _kwargs_tangents = torch.split(y, [y_dim, y_dim, const_total_dim])
                    _kwargs_tangents = torch.split(_kwargs_tangents, const_dims)
                    _kwargs_tangents = [i.view(j) for j,i in zip(const_shapes, _kwargs_tangents)]
                else:
                    _y, _tan = torch.split(y, [y_dim, y_dim])
                    _kwargs_tangents = []
                _y, _tan = _y.view(y_shape), _tan.view(y_shape)
                _dydt, _dtandt = torch.autograd.functional.jvp(wrapped_rhs, (t, _y, *[kwargs[key] for key in tensor_constants]), (torch.zeros_like(t), _tan, *_kwargs_tangents))
                ret_dydt = torch.cat([
                    _dydt.view(-1),
                    _dtandt.view(-1),
                    *[torch.zeros_like(i.view(-1)) for i in _kwargs_tangents]
                ], dim=-1)
                return ret_dydt
            
            tan_t0 = t_span[0]
            time_tangents = torch.zeros_like(evaluation_times)
            if t0_tangent is not None:
                time_tangents[0] = t0_tangent
            state_tangents = torch.zeros_like(evaluation_states)
            if y0_tangent is not None:
                state_tangents[...,0] = y0_tangent
            
            rhs_at_t0 = wrapped_rhs(
                t_span[0], y0.view(y_shape), *[constants[key] for key in tensor_constants]
            )
            rhs_at_tf = wrapped_rhs(
                t_span[1], evaluation_states[...,-1].view(y_shape), *[constants[key] for key in tensor_constants]
            )
            
            augmented_y = torch.cat([
                y0.view(-1),
                (y0_tangent - rhs_at_t0*time_tangents[0]).view(-1),
            ], dim=-1)

            if len(tensor_constants) > 0:
                augmented_y = torch.cat([
                    augmented_y,
                    *[constants_tangents[key].view(-1) for key in tensor_constants]
                ], dim=-1)
            
            options["atol"], options["rtol"] = ctx.atol, ctx.rtol
            options["atol"] = torch.cat([
                torch.ones_like(y0.view(-1))*options["atol"],
                torch.ones_like(augmented_y[y_dim:])*torch.inf
            ], dim=-1)
            options["rtol"] = torch.cat([
                torch.ones_like(y0.view(-1))*options["rtol"],
                torch.ones_like(augmented_y[y_dim:])*torch.inf
            ], dim=-1)
            res = torch_solve_ivp(augmented_forward_fn, t_span=(tan_t0, t_span[1]), y0=augmented_y, method=method, events=events, kwargs={key: constants[key] for key in tensor_constants}, **options)
            if const_total_dim is not None:
                state_tangents = torch.split(res.y, [y_dim, y_dim, const_total_dim])[1].reshape(*y_shape, -1).clone().contiguous()
            else:
                state_tangents = torch.split(res.y, [y_dim, y_dim])[1].reshape(*y_shape, -1).clone().contiguous()
        
            if tf_tangent is not None:
                time_tangents[-1] = tf_tangent
            state_tangents[...,-1] = state_tangents[...,-1] + rhs_at_tf*time_tangents[-1]
            
            return time_tangents.contiguous(), state_tangents.contiguous(), None
    

    soln = WrappedSolveIVP.apply(*flattened_parameters)
    ode_sys_soln = soln[2]
    return OdeResult(
        t=soln[0], y=soln[1], sol=ode_sys_soln.ode_system.sol, t_events=ode_sys_soln.ode_system.events, 
        y_events=ode_sys_soln.ode_system.events, nfev=ode_sys_soln.ode_system.nfev, njev=ode_sys_soln.ode_system.njev,
        status=ode_sys_soln.ode_system.integration_status, message=ode_sys_soln.ode_system.integration_status,
        success=ode_sys_soln.ode_system.success, ode_system=ode_sys_soln.ode_system
    )
