The software is designed to measure the market risk of an arbitrary portfolio which consists of several stocks invested in a certain period. The market risk metrics used include value-at-risk (VaR) and expected shortfall (ES), and three types of models are built in for evaluation, 1) parametric model, 2) historical model and 3) Monte Carlo model. Also, through the backtest, users can see which method is the most suitable for a certain portfolio.

Architecture and directory
1. Utils (static): Functions and class objects that define parametric, historical, and Monte Carlo VaR/ES computation are managed under ‘utils’ directory. 
2. Data (user control): As introduced in (a), configuration files and historical data are saved here. 
3. Output (static): Figures are saved under ‘figures’ in .png format. VaR/ES results and backtest results are saved into excel files categorized by models. Note that to keep the simplicity of output, only VaR/ES results are exported, while intermediate outputs such as μ and σ of portfolio and each stock, can be retrieved from class attributes.
4. Backtest:
5. System.py (user control): The main program of the risk management system to be executed. Specifically, users are required to run system.py after providing configuration parameters. 

Portfolios ready for backtest now: (including equal weight and custom weight)
1. Parametric model: 
   - Long-only (1) - gbm and normal
   - Short-only (2) - gbm and normal

2. Historical model:
   - Long-only (1)
   - Short-only (2)

3. Monte Carlo model:
   - Long-only (1) - gbm
   - Short-only (2) - gbm
   - Long only + put option (one stock, one option with implied vol provided)
