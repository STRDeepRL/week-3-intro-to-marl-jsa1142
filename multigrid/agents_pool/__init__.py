from multigrid.agents_pool.JamesStankowicz_policies.Default_policy import Policy as DefaultPolicy
from multigrid.agents_pool.JamesStankowicz_policies.Eliminator_policy import Policy as EliminatorPolicy
from multigrid.agents_pool.JamesStankowicz_policies.PickUpper_policy import Policy as PickUpperPolicy

SubmissionPolicies = {
    "default": DefaultPolicy,
    "eliminator": EliminatorPolicy,
    "pickupper": PickUpperPolicy,
}
