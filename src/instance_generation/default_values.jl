# --- Default values for structure attributes ---

# default objective weights
const default_delay_cost = 2.0
const default_vehicle_cost = 1000.0

const default_width = 50
const default_αᵥ_low = 1.2   # used for drawing random tasks
const default_αᵥ_high = 1.6  # used for drawing random tasks
const default_first_begin_time = 60.0 * 6  # 06h
const default_last_begin_time = 60.0 * 20  # 20h
const default_nb_tasks = 10

const default_district_width = 10
const default_random_inter_area_factor = LogNormal(0.02, 0.05)

const ZERO_UNIFORM = LogNormal(-Inf, 1.) # always returns 0

const default_district_μ = Uniform(0.8, 1.2)
const default_district_σ = Uniform(0.4, 0.6)

const default_task_μ = Uniform(1.0, 3.0)
const default_task_σ = Uniform(0.8, 1.2)

const default_nb_scenarios = 1
