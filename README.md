# Killing Decoherences with Collisions

This project performs physics simulations to study the **Petz Recovery Map (PRM)** $\Lambda$ on quantum noise channels (specifically Amplitude Damping, Dephasing, and Bitflip) using Collision Models.

The workflow revolves around evolving quantum states $\rho$ through a noise channel, attempting to reverse the noise using a Petz Recovery Map, and measuring the resulting Fidelity to determine if the quantum state was successfully preserved.

## 1. Prerequisites & Installation

### Step 1: Install Julia
The recommended way to install and manage Julia versions is via [juliaup](https://github.com/JuliaLang/juliaup).
### Step 2: Global Tools
It is highly recommended to have `Revise.jl` installed in your global environment. 
This allows you to modify code in `src/` and see changes instantly without restarting the REPL.
Open a terminal and run:
```bash
julia -e 'using Pkg; Pkg.add("Revise")'
```
## 2. Setup
Clone the repository and navigate into the folder:
```bash 
git clone https://github.com/fedesss98/shooting-decoherences.git
cd shooting-decoherences
```
Here you'll find the functions to perform experiments in the Julia REPL.
First, instantiate the project. 
This downloads all required dependencies specified in `Project.toml`:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
## 3. Workflow
This project is designed to be run interactively via the Julia REPL. 
We use a specific startup command to:
- Activate the project environment;
- Load `Revise` (for hot-reloading code);
- Include the main simulation script (`scripts/main.jl`) using `includet` (track changes).
### Fast StartupRun 
this single command in your terminal from the project root:
```bash 
julia --project=. -e "using Revise; includet(\"scripts/main.jl\")" -i
```
*Note: The `-i` flag keeps the REPL open in interactive mode after the script loads.*

## 4. Running Experiments
Once inside the REPL (you should see the `julia>` prompt), you can run simulations using the functions defined in `scripts/main.jl`.

### Basic Experiment
Run a single simulation comparing free evolution vs. Petz recovery. 
You can override default parameters (`gamma`, `n_qubits`, etc.) via keyword arguments.
```julia
# Run a simple bitflip experiment on 3 qubits
f1, f2, states, avg_fid = run_experiment("bitflip"; n_qubits=3, beta=0.5)
```
### Batch Experiments (Fig 3 of *The Reference*)
To generate data for "Average Infidelity vs $n\beta$", use `run_fig3_experiment`.
**Pro Tip**: Use `begin ... end` blocks in the REPL. 
This allows you to paste multi-line configurations easily without the REPL executing lines prematurely.
```julia
begin
    # Define plot points (x-axis values for n*beta)
    points = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    # Run experiment for different qubit sizes
    run_fig3_experiment(
        points, 
        [3, 4, 5],       # List of N_qubits to test
        "damping";       # Noise model
        time=0.8,        # Time tau
        plot=true        # Automatically generate the plot
    )
end
```
## Iterating Over Parameters
If you need to sweep over multiple beta values and qubit counts and save the data to JSON:
```julia 
begin
    betas_to_test = range(0.1, 1.0, length=5)
    qubits_to_test = [2, 4]
    
    iterate_exps(
        betas_to_test, 
        qubits_to_test; 
        noise_model="dephasing"
    )
end
```

## 5. Project Structure
- `scripts/`: Contains execution scripts.
    - `main.jl`: The primary entry point containing experiment runners (`run_experiment`) and plotting logic.
- `src/`: The core library logic.
    - `PetzMaps.jl`: Implementation of the recovery map math.
    - `operators.jl` & `quantum_states.jl`: Definitions of quantum channels and state preparations.
    - `DecoKiller.jl`: Main module wrapper.
- `visualization/`: Folder where plots (.png, .pdf) are saved.

## 6. Visualization
Plots are automatically saved to the `../visualization/` directory relative to where the script is run. 
Ensure this directory exists or the script may attempt to create it (depending on write permissions).

---
