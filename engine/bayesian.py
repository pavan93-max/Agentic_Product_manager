import os
import sys

os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float32,optimizer=fast_run,mode=FAST_RUN'
os.environ['PYTENSOR_COMPILE_MODE'] = 'FAST_RUN'
os.environ['PYMC_COMPILE_MODE'] = 'fast_run'

try:
    import pytensor
    pytensor.config.mode = 'FAST_RUN'
    pytensor.config.optimizer = 'fast_run'
    pytensor.config.cxx = ''
    pytensor.config.gcc__cxxflags = ''
    pytensor.config.linker__c = ''
    pytensor.config.linker__cxx = ''
except:
    pass

import pymc as pm
import numpy as np
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)


def bayesian_ab_test(
    control: np.ndarray,
    treatment: np.ndarray,
    samples: int = 2000,
    tune: int = 1000
) -> Dict[str, Union[float, list]]:
    if len(control) == 0 or len(treatment) == 0:
        raise ValueError("Control and treatment arrays must not be empty")
    if not isinstance(control, np.ndarray):
        control = np.array(control, dtype=np.float32)
    if not isinstance(treatment, np.ndarray):
        treatment = np.array(treatment, dtype=np.float32)
    
    control = control.astype(np.float32)
    treatment = treatment.astype(np.float32)
    
    logger.info(f"Running Bayesian A/B test: control n={len(control)}, treatment n={len(treatment)}")
    
    try:
        import pytensor
        old_mode = pytensor.config.mode
        old_optimizer = pytensor.config.optimizer
        
        pytensor.config.mode = 'FAST_RUN'
        pytensor.config.optimizer = 'fast_run'
        pytensor.config.cxx = ''
        pytensor.config.gcc__cxxflags = ''
        
        try:
            with pm.Model():
                p_c = pm.Beta("p_c", alpha=1, beta=1)
                p_t = pm.Beta("p_t", alpha=1, beta=1)
                pm.Bernoulli("obs_c", p=p_c, observed=control)
                pm.Bernoulli("obs_t", p=p_t, observed=treatment)
                
                try:
                    trace = pm.sample(
                        samples, 
                        tune=tune, 
                        progressbar=False, 
                        return_inferencedata=True,
                        compute_convergence_checks=False,
                        target_accept=0.8,
                        cores=1,
                        chains=1
                    )
                except Exception as compile_error:
                    if "Compilation failed" in str(compile_error) or "64-bit mode" in str(compile_error):
                        logger.warning("PyMC compilation failed (likely MinGW 32-bit issue). Using analytical approximation...")
                        return _analytical_bayesian_approximation(control, treatment)
                    else:
                        raise
        finally:
            try:
                pytensor.config.mode = old_mode
                pytensor.config.optimizer = old_optimizer
            except:
                pass
    except Exception as e:
        error_msg = str(e)
        if "Compilation failed" in error_msg or "64-bit mode" in error_msg or "lazylinker" in error_msg.lower():
            logger.warning(f"PyMC compilation issue detected: {error_msg[:200]}. Using analytical approximation...")
            return _analytical_bayesian_approximation(control, treatment)
        else:
            logger.warning(f"PyMC sampling failed: {error_msg[:200]}. Using analytical approximation...")
            return _analytical_bayesian_approximation(control, treatment)
    
    lift = trace.posterior["p_t"] - trace.posterior["p_c"]
    result = {
        "lift_mean": float(lift.mean()),
        "prob_treatment_better": float((lift > 0).mean()),
        "ci_95": [
            float(lift.quantile(0.025)),
            float(lift.quantile(0.975))
        ]
    }
    logger.info(f"Bayesian analysis complete: lift={result['lift_mean']:.4f}, "
                f"P(treatment>control)={result['prob_treatment_better']:.4f}")
    return result


def _analytical_bayesian_approximation(control: np.ndarray, treatment: np.ndarray) -> Dict[str, Union[float, list]]:
    try:
        from scipy import stats
    except ImportError:
        logger.error("scipy not available for analytical approximation")
        raise RuntimeError("PyMC failed and scipy is not available for fallback")
    
    control_successes = int(control.sum())
    control_failures = len(control) - control_successes
    treatment_successes = int(treatment.sum())
    treatment_failures = len(treatment) - treatment_successes
    
    alpha_c = 1 + control_successes
    beta_c = 1 + control_failures
    alpha_t = 1 + treatment_successes
    beta_t = 1 + treatment_failures
    
    p_c_dist = stats.beta(alpha_c, beta_c)
    p_t_dist = stats.beta(alpha_t, beta_t)
    
    p_c_mean = p_c_dist.mean()
    p_t_mean = p_t_dist.mean()
    lift_mean = p_t_mean - p_c_mean
    
    samples = 10000
    p_c_samples = p_c_dist.rvs(samples)
    p_t_samples = p_t_dist.rvs(samples)
    lift_samples = p_t_samples - p_c_samples
    
    prob_treatment_better = (lift_samples > 0).mean()
    ci_95 = [np.percentile(lift_samples, 2.5), np.percentile(lift_samples, 97.5)]
    
    return {
        "lift_mean": float(lift_mean),
        "prob_treatment_better": float(prob_treatment_better),
        "ci_95": [float(ci_95[0]), float(ci_95[1])]
    }
