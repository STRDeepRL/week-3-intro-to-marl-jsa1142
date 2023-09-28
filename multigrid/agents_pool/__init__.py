from multigrid.agents_pool.JamesStankowicz_policies.Default_policy import DefaultPolicy
from multigrid.agents_pool.JamesStankowicz_policies.Eliminator_policy import EliminatorPolicy
from multigrid.agents_pool.JamesStankowicz_policies.PickUpper_policy import PickUpperPolicy

SubmissionPolicies = {
    "default": DefaultPolicy,
    "eliminator": EliminatorPolicy,
    "pickupper": PickUpperPolicy,
}
