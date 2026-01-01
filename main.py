import json
import re
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from crew.crew import experiment_crew
from engine.simulator import simulate_users
from engine.bayesian import bayesian_ab_test
from engine.decision_rule import decide
from engine.schemas import ExperimentDesign, BayesianResult
from engine.memory import log_experiment

import sys
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()


def fix_json_string(json_str: str) -> str:
    json_str = json_str.strip()
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
    json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
    lines = json_str.split('\n')
    fixed_lines = []
    in_string = False
    escape_next = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped or re.search(r"^\s*//", line) or re.search(r"^\s*#", line):
            continue
        
        for i, char in enumerate(line):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
            elif not in_string and char == "'":
                line = line[:i] + '"' + line[i+1:]
                in_string = True
        
        fixed_lines.append(line)
    
    json_str = '\n'.join(fixed_lines)
    return json_str


def extract_json_from_text(text: str) -> Optional[str]:
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        candidate = json_match.group(1)
        try:
            json.loads(candidate)
            return candidate
        except:
            pass
    
    brace_count = 0
    start_idx = -1
    candidates = []
    
    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                candidate = text[start_idx:i+1]
                try:
                    json.loads(candidate)
                    candidates.append((len(candidate), candidate))
                except:
                    pass
                start_idx = -1
    
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return None


def parse_experiment_design(raw_output: str) -> Dict[str, Any]:
    if not raw_output or not isinstance(raw_output, str):
        raise ValueError("Raw output must be a non-empty string")
    
    experiment_design_raw = extract_json_from_text(raw_output)
    if not experiment_design_raw:
        logger.error(f"Could not extract JSON. Raw output preview: {raw_output[:500]}")
        raise ValueError("Could not extract JSON from experiment design output")
    
    experiment_design = None
    json_errors = []
    
    strategies = [
        ("Standard JSON", lambda x: json.loads(x)),
        ("Fixed JSON", lambda x: json.loads(fix_json_string(x))),
    ]
    
    for strategy_name, strategy_func in strategies:
        try:
            experiment_design = strategy_func(experiment_design_raw)
            logger.info(f"Successfully parsed JSON using {strategy_name}")
            break
        except Exception as e:
            json_errors.append(f"{strategy_name}: {str(e)}")
            continue
    
    if experiment_design is None:
        try:
            import json5
            experiment_design = json5.loads(experiment_design_raw)
            logger.info("Successfully parsed JSON using JSON5")
        except ImportError:
            json_errors.append("JSON5 not available")
        except Exception as e:
            json_errors.append(f"JSON5: {str(e)}")
    
    if experiment_design is None:
        logger.error(f"Failed to parse JSON. Errors: {json_errors}")
        logger.error(f"Raw JSON (first 1000 chars): {experiment_design_raw[:1000]}")
        raise ValueError(f"Invalid JSON in experiment design. Errors: {'; '.join(json_errors)}")
    
    if not isinstance(experiment_design, dict):
        raise ValueError(f"Experiment design must be a dictionary, got {type(experiment_design)}")
    
    control_variant = None
    treatment_variant = None
    
    if 'variants' in experiment_design:
        variants = experiment_design['variants']
        if isinstance(variants, dict):
            control_keys = ['control', 'Control', 'control_variant', 'Control Variant', 'baseline', 'Baseline', 'A', 'variant_a']
            treatment_keys = ['treatment', 'Treatment', 'treatment_variant', 'Treatment Variant', 'variant', 'Variant', 'B', 'variant_b']
            
            for key in control_keys:
                if key in variants:
                    control_variant = variants[key]
                    break
            
            for key in treatment_keys:
                if key in variants:
                    treatment_variant = variants[key]
                    break
            
            if not control_variant or not treatment_variant:
                variant_keys = [k for k in variants.keys() if variants[k] is not None]
                if len(variant_keys) >= 2:
                    control_variant = variants.get(variant_keys[0])
                    treatment_variant = variants.get(variant_keys[1])
                elif len(variant_keys) == 1:
                    control_variant = variants.get(variant_keys[0])
                    treatment_variant = {}
        elif isinstance(variants, list):
            if len(variants) >= 2:
                control_variant = variants[0] if variants[0] is not None else {}
                treatment_variant = variants[1] if variants[1] is not None else {}
            elif len(variants) == 1:
                if isinstance(variants[0], dict):
                    variant_dict = variants[0]
                    control_keys = ['control', 'Control', 'baseline', 'Baseline']
                    treatment_keys = ['treatment', 'Treatment', 'variant', 'Variant']
                    for key in control_keys:
                        if key in variant_dict:
                            control_variant = variant_dict[key]
                            break
                    for key in treatment_keys:
                        if key in variant_dict:
                            treatment_variant = variant_dict[key]
                            break
    
    if not control_variant:
        control_variant = (
            experiment_design.get('control') or 
            experiment_design.get('control_variant') or 
            experiment_design.get('baseline') or
            experiment_design.get('variant_a') or
            {}
        )
    if not treatment_variant:
        treatment_variant = (
            experiment_design.get('treatment') or 
            experiment_design.get('treatment_variant') or 
            experiment_design.get('variant') or
            experiment_design.get('variant_b') or
            {}
        )
    
    if not isinstance(control_variant, dict):
        control_variant = {"value": control_variant} if control_variant is not None else {}
    if not isinstance(treatment_variant, dict):
        treatment_variant = {"value": treatment_variant} if treatment_variant is not None else {}
    
    sample_size = (
        experiment_design.get('sample_size') or 
        experiment_design.get('sample_size_per_variant') or
        experiment_design.get('sample_size_estimation') or
        experiment_design.get('sample_size_per_group') or
        experiment_design.get('n_per_group') or
        experiment_design.get('n')
    )
    
    if isinstance(sample_size, dict):
        sample_size = (
            sample_size.get('per_variant') or 
            sample_size.get('per_group') or 
            sample_size.get('value') or
            sample_size.get('n')
        )
    if isinstance(sample_size, str):
        sample_size = re.sub(r'[^\d]', '', sample_size)
        try:
            sample_size = int(sample_size) if sample_size else None
        except (ValueError, TypeError):
            sample_size = None
    if isinstance(sample_size, (float, int)):
        sample_size = int(sample_size)
    else:
        sample_size = None
    
    metric_raw = (
        experiment_design.get('metric') or 
        experiment_design.get('primary_metric') or 
        experiment_design.get('primary_evaluation_metric') or
        experiment_design.get('evaluation_metric') or
        experiment_design.get('target_metric') or
        'conversion_rate'
    )
    
    if isinstance(metric_raw, dict):
        metric = (
            metric_raw.get('name') or 
            metric_raw.get('metric') or 
            metric_raw.get('type') or 
            metric_raw.get('key') or
            str(metric_raw)
        )
    elif isinstance(metric_raw, str):
        metric = metric_raw.strip()
    else:
        metric = str(metric_raw) if metric_raw else 'conversion_rate'
    
    if not metric:
        metric = 'conversion_rate'
    
    if not control_variant or not treatment_variant:
        available_keys = list(experiment_design.keys())
        logger.error(f"Available experiment_design keys: {available_keys}")
        if 'variants' in experiment_design:
            logger.error(f"Variants structure: {type(experiment_design['variants'])} - {str(experiment_design['variants'])[:500]}")
        raise ValueError(
            f"Missing required keys in experiment_design. "
            f"Required: control/control_variant, treatment/treatment_variant. "
            f"Available keys: {available_keys}"
        )
    
    if not sample_size or sample_size <= 0:
        sample_size = 1000
        logger.warning(f"Sample size not found or invalid, using default: {sample_size}")
    
    return {
        'control': control_variant,
        'treatment': treatment_variant,
        'sample_size': sample_size,
        'metric': metric
    }


def main() -> None:
    logger.info("Starting autonomous experimentation pipeline...")
    try:
        logger.info("Step 1: Running CrewAI agents to generate experiment design...")
        crew_output = experiment_crew.kickoff()
        
        if not crew_output or not hasattr(crew_output, 'tasks_output'):
            raise ValueError("Invalid crew output: missing tasks_output")
        
        if len(crew_output.tasks_output) < 3:
            raise ValueError(f"Insufficient tasks completed. Expected 3, got {len(crew_output.tasks_output)}")
        
        idea = crew_output.tasks_output[0].raw if crew_output.tasks_output[0].raw else "No idea generated"
        hypothesis = crew_output.tasks_output[1].raw if crew_output.tasks_output[1].raw else "No hypothesis generated"
        experiment_design_raw = crew_output.tasks_output[2].raw if crew_output.tasks_output[2].raw else ""
        
        if not experiment_design_raw:
            raise ValueError("Experiment design output is empty")
        
        logger.info("Experiment design generated")
        logger.info("Step 2: Parsing experiment design...")
        normalized_experiment_design = parse_experiment_design(experiment_design_raw)
        
        try:
            experiment = ExperimentDesign(**normalized_experiment_design)
        except Exception as e:
            logger.error(f"Failed to create ExperimentDesign: {e}")
            logger.error(f"Normalized design: {normalized_experiment_design}")
            raise ValueError(f"Invalid experiment design structure: {e}") from e
        
        logger.info(f"Experiment design parsed: {experiment.metric}, n={experiment.sample_size}")
        logger.info("Step 3: Simulating user behavior...")
        
        try:
            control_data = simulate_users(experiment.control, experiment.sample_size)
            treatment_data = simulate_users(experiment.treatment, experiment.sample_size)
        except Exception as e:
            logger.error(f"User simulation failed: {e}")
            raise ValueError(f"Failed to simulate users: {e}") from e
        
        if len(control_data) == 0 or len(treatment_data) == 0:
            raise ValueError(f"Simulation produced empty data: control={len(control_data)}, treatment={len(treatment_data)}")
        
        logger.info(f"Simulated {len(control_data)} control and {len(treatment_data)} treatment users")
        logger.info("Step 4: Running Bayesian A/B test analysis...")
        
        try:
            bayes_result = bayesian_ab_test(
                control=control_data,
                treatment=treatment_data
            )
        except Exception as e:
            logger.error(f"Bayesian analysis failed: {e}")
            raise ValueError(f"Bayesian analysis error: {e}") from e
        
        if not isinstance(bayes_result, dict) or 'lift_mean' not in bayes_result:
            raise ValueError(f"Invalid Bayesian result structure: {bayes_result}")
        
        try:
            bayes = BayesianResult(**bayes_result)
        except Exception as e:
            logger.error(f"Failed to create BayesianResult: {e}")
            logger.error(f"Bayesian result: {bayes_result}")
            raise ValueError(f"Invalid Bayesian result: {e}") from e
        
        logger.info("Bayesian analysis complete")
        logger.info("Step 5: Making autonomous decision...")
        
        try:
            final_decision = decide(bayes_result)
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            final_decision = "ITERATE"
            logger.warning(f"Using fallback decision: {final_decision}")
        
        if final_decision not in ["SHIP", "ROLLBACK", "ITERATE"]:
            logger.warning(f"Invalid decision '{final_decision}', defaulting to ITERATE")
            final_decision = "ITERATE"
        
        logger.info(f"Decision: {final_decision}")
        logger.info("Step 6: Logging experiment results...")
        
        try:
            log_experiment({
                "experiment": experiment.model_dump(),
                "bayesian_result": bayes.model_dump(),
                "decision": final_decision
            })
        except Exception as e:
            logger.warning(f"Failed to log experiment: {e}. Continuing...")
        
        logger.info("Experiment logged")
        print("\n" + "="*60)
        print("AUTONOMOUS EXPERIMENT RESULT")
        print("="*60)
        print(f"\nIdea: {idea}")
        print(f"\nHypothesis: {hypothesis}")
        print(f"\nBayesian Analysis:")
        print(f"   - Posterior Lift: {bayes_result['lift_mean']*100:.2f}%")
        print(f"   - P(Treatment > Control): {bayes_result['prob_treatment_better']:.4f}")
        print(f"   - 95% Credible Interval: [{bayes_result['ci_95'][0]:.4f}, {bayes_result['ci_95'][1]:.4f}]")
        print(f"\nDecision: {final_decision}")
        print("="*60 + "\n")
        logger.info("Pipeline execution completed successfully")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
