
# Using Qwen2.5-VL-7B-Instruct-GGUF for Cryptocurrency Trading Chart Analysis
---

#### Abstract
This technical report details the implementation of a system that captures BTC-USDT trading charts, calculates technical indicators, and analyzes the charts using the Qwen2.5-VL-7B-Instruct-GGUF model from Hugging Face. The system leverages the Binance API for real-time data, Matplotlib and `mplfinance` for chart generation, and Hugging Face's image-to-text model for analysis.

#### Introduction
Cryptocurrency trading involves analyzing market trends and patterns to make informed decisions. Technical analysis plays a crucial role in this process by using statistical tools to identify trends, support and resistance levels, and potential future price movements. This project aims to automate the process of generating trading charts and analyzing them using advanced machine learning models.

#### System Architecture

1. **Data Fetching**:
   - **Binance API**: Fetches historical price data for BTC-USDT.
   - **Intervals**: Configurable intervals (e.g., 1 minute) and data points (e.g., 60 data points).

2. **Technical Indicators Calculation**:
   - **Simple Moving Average (SMA)** and **Exponential Moving Average (EMA)**: Smooth out price data to identify trends.
   - **Relative Strength Index (RSI)**: Measures the speed and change of price movements.
   - **Moving Average Convergence Divergence (MACD)**: Identifies potential buy and sell signals.
   - **Candlestick Patterns**: Detects specific candlestick patterns such as Hammer and Engulfing.

3. **Chart Generation**:
   - **Matplotlib and `mplfinance`**: Generates candlestick charts with additional subplots for technical indicators.
   - **Annotations**: Highlights detected candlestick patterns.

4. **Image-to-Text Analysis**:
   - **Qwen2.5-VL-7B-Instruct-GGUF Model**: Analyzes the generated charts using Hugging Face's image-to-text model.
   - **API Endpoint**: Sends the chart images to the Hugging Face inference endpoint for analysis.

5. **Scheduling**:
   - **Schedule Library**: Automates the process to run every 10 minutes.

#### Implementation Details

1. **Environment Setup**:
   - **Dependencies**: Installed using `requirements.txt`.
   - **Environment Variables**: Stored in `.env` for security.

2. **Data Fetching**:
   - **Function**: `fetch_historical_data` in `utils.py`.
   - **Parameters**: Symbol (`BTCUSDT`), interval (`1m`), and limit (`60`).

3. **Technical Indicators Calculation**:
   - **Function**: `calculate_indicators` in `utils.py`.
   - **Indicators**: SMA, EMA, RSI, MACD, Hammer, and Engulfing patterns.

4. **Chart Generation**:
   - **Function**: `plot_chart` in `utils.py`.
   - **Subplots**: Candlestick chart, SMA, EMA, MACD, RSI, and annotations for candlestick patterns.

5. **Image-to-Text Analysis**:
   - **Function**: `analyze_chart_with_llm` in `model.py`.
   - **API Request**: Sends the chart image to the Hugging Face endpoint.
   - **Response Handling**: Parses the generated text analysis.

6. **Scheduling**:
   - **Function**: `capture_and_analyze_chart` in `model.py`.
   - **Schedule**: Runs every 10 minutes using the `schedule` library.

#### Results and Discussion

The system successfully generates BTC-USDT trading charts with technical indicators and analyzes them using the Qwen2.5-VL-7B-Instruct-GGUF model. The following are key observations:

1. **Chart Accuracy**:
   - The charts accurately represent the historical price data and technical indicators.
   - Candlestick patterns are correctly annotated.

2. **Model Performance**:
   - The Qwen2.5-VL-7B-Instruct-GGUF model provides insightful analysis of the charts.
   - The analysis includes trend predictions, potential buy/sell signals, and other relevant information.

3. **Automation**:
   - The scheduling mechanism ensures that the analysis is performed periodically, making it suitable for continuous monitoring.

#### Conclusion

This project demonstrates the effectiveness of combining real-time data fetching, technical analysis, and advanced machine learning models for cryptocurrency trading chart analysis. The system automates the process, providing valuable insights that can aid traders in making informed decisions.

#### Future Work

1. **Enhanced Indicators**:
   - Incorporate additional technical indicators for more comprehensive analysis.

2. **User Interface**:
   - Develop a web-based interface for real-time visualization and interaction.

3. **Alerts**:
   - Implement alert mechanisms to notify users of significant events or patterns.

4. **Backtesting**:
   - Conduct backtesting to evaluate the accuracy and profitability of the analysis.

#### References

- Binance API Documentation: [https://binance-docs.github.io/apidocs/spot/en/](https://binance-docs.github.io/apidocs/spot/en/)
- Hugging Face Model Documentation: [https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF)
- Ta Library: [https://technical-analysis-library-in-python.readthedocs.io/](https://technical-analysis-library-in-python.readthedocs.io/)
- Mplfinance Library: [https://github.com/matplotlib/mplfinance](https://github.com/matplotlib/mplfinance)

