# Random Forest Stock Trading Learner

This project investigates the **in-sample and out-of-sample trading performance** of a manual strategy and a **Random Forest classification-based learner**, both built using three technical indicators:  
- **Bollinger Bands**  
- **Relative Strength Index (RSI)**  
- **Stochastic Oscillator**  

In both the manual strategy and Random Forest learner, we used the same starting cash of $100,000. Allowable positions are 1000 shares long, 1000 shares short, or 0 shares. We can trade up to 2000 shares at a time, with unlimited leverage, assum-
ing commission of $9.95 and trade impact of 0.005 (0.5%). JPM (JPMorgan Chase & Co.) is the only symbol used. The in-sample time period is from January 1, 2008 to December 31, 2009, and the out-of-sample period is from January 1, 2010 to December 31, 2011.

Subsequently, we also explore the effects of different values of market impact (0.0005, 0.005, 0.05) on **cumulative returns** and the **number of trades** of the Random Forest learner during the in-sample time period.

---

## Instructions to Run the Project Locally

### Prerequisites
Ensure you have **Python** and the following libraries installed:  
- `NumPy`  
- `Pandas`  
- `Matplotlib`  
- `SciPy`  

Alternatively, if you have **Anaconda/Miniconda** installed:  
1. Create the environment with the provided `environment.yml` file:  
   ```bash
   conda env create --file environment.yml
2. Activate the environment:
   ```bash
   conda activate ml4t

### Running the Project
1. Excecute the project script:
   ```bash
   python testproject.py

### Viewing Generated Outputs
Performance charts: Located in the `images` folder
In-Sample performance statistics: `Performance - In-Sample for JPM.txt`
Out-of-Sample performance statistics: `Performance - Out-of-Sample for JPM.txt`
Market impact on in-sample performance: `Market Impact - In-Sample for JPM.txt`