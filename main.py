import asyncio
from typing import List, Dict
import sys
import os

import json
from aiohttp import ClientSession, ClientTimeout
from asknews_sdk import AskNewsSDK
import aiohttp
import argparse
import logging
import re
import random
from datetime import datetime
from typing import Literal
import time
import traceback

ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)

def write(x):
    print(x)

class FallTemplateBot2025(ForecastBot):
    """
    This is a copy of the template bot for Fall 2025 Metaculus AI Tournament.
    This bot is what is used by Metaculus in our benchmark, but is also provided as a template for new bot makers.
    This template is given as-is, and though we have covered most test cases
    in forecasting-tools it may be worth double checking key components locally.

    Main changes since Q2:
    - An LLM now parses the final forecast output (rather than programmatic parsing)
    - Added resolution criteria and fine print explicitly to the research prompt
    - Previously in the prompt, nothing about upper/lower bound was shown when the bounds were open. Now a suggestion is made when this is the case.
    - Support for nominal bounds was added (i.e. when there are discrete questions and normal upper/lower bounds are not as intuitive)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ones.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLMto intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions for the
    MiniBench and Seasonal AIB tournaments. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            # "researcher": "asknews/deep-research/low",
            "researcher" = "openrouter/perplexity/sonar-reasoning"
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/deep-research/low":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "model_name").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _model_number = 0 # used to iterate through the models
    _number_of_models = 4 # number of all models we are calling
    """
    model0="openrouter/google/gemini-2.5-pro-preview-03-25",
    model1="openrouter/anthropic/claude-3.7-sonnet",
    model2="openrouter/deepseek/deepseek-chat-v3-0324",
    model3="openrouter/x-ai/grok-3-beta",
    model4="openrouter/openai/gpt-4.1",
    model5="openrouter/google/gemini-2.5-flash-preview",
    """

    async def call_asknews_latest(self, question: str) -> str:
      """
      Use the AskNews `news` endpoint to get news context for your query.
      The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
      """
      try:
          ask = AskNewsSDK(
              client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=set(["news"])
          )

          async with aiohttp.ClientSession() as session:
              # Create tasks for both API calls
              hot_task = asyncio.create_task(asyncio.to_thread(ask.news.search_news,
                  query=question,
                  n_articles=8,
                  return_type="both",
                  strategy="latest news"
              ))
              hot_response = await asyncio.gather(hot_task)

#          async with aiohttp.ClientSession() as session:
#              historical_task = asyncio.create_task(asyncio.to_thread(ask.news.search_news,
#                  query=question,
#                  n_articles=8,
#                  return_type="string",
#                  strategy="news knowledge"
#              ))
#
#              # Wait for both tasks to complete
#              historical_response = await asyncio.gather(historical_task)

          # historical_articles = historical_response.as_dicts
          historical_articles = None
          formatted_articles = "Here are the relevant news articles:\n\n"
          for response in hot_response:
            hot_articles = response.as_dicts

            if hot_articles:
              hot_articles = [article.__dict__ for article in hot_articles]
              hot_articles = sorted(hot_articles, key=lambda x: x["pub_date"], reverse=True)

              for article in hot_articles:
                  pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                  formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

          if historical_articles:
              historical_articles = [article.__dict__ for article in historical_articles]
              historical_articles = sorted(
                  historical_articles, key=lambda x: x["pub_date"], reverse=True
              )

              for article in historical_articles:
                  pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                  formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

          if not hot_articles and not historical_articles:
              formatted_articles += "No articles were found.\n\n"
              return formatted_articles

          return formatted_articles
      except Exception as e:
          write(f"[call_asknews] Error: {str(e)}")
          return f"Error retrieving news articles: {str(e)}"
    
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
You are an assistant to a superforecaster.
The superforecaster will give you a question they intend to forecast on.
Your goal is to provide a concise but detailed research briefing based on prior trends and the *latest* available information.
Focus on identifying and summarizing the most *critical* developments, events, and expert opinions directly relevant to the question's potential resolution.
Structure your rundown clearly, perhaps using bullet points or distinct paragraphs for different key angles.
Explicitly address how both long-term trends and the *current* information weighs towards a "Yes" or "No" resolution, citing the key pieces of information supporting that assessment.
Do NOT produce a forecast yourself or assign probabilities. Stick to summarizing the current situation and its implications for the resolution as it stands now.

The question from the superforecaster is: {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif researcher == "asknews/news-summaries":
                research = await self.call_asknews_latest(question.question_text)
                #research = await AskNewsSearcher().get_formatted_news_async(
                #    question.question_text
                #)
            elif researcher == "asknews/deep-research/medium-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews", "google"],
                    search_depth=2,
                    max_depth=4,
                )
            elif researcher == "asknews/deep-research/high-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews", "google"],
                    search_depth=4,
                    max_depth=6,
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    def _random_model(self) -> str:
        return "model" + str(random.randint(0,self._number_of_models-1))

    def _next_model(self) -> str:
        _model_name = "model" + str(self._model_number)
        self._model_number = (self._model_number + 1) % self._number_of_models
        return _model_name
        
    async def summarize_research(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        if not self.enable_summarize_research:
            return "Summarize research was disabled for this run"

        try:
            logger.info(f"Summarizing research for question: {question.page_url}")
            model = self.get_llm("summarizer", "llm")
            prompt = clean_indents(
                f"""
                Please summarize the following research in 3-4 paragraphs. The research tries to help answer the following question:
                {question.question_text}

                Only summarize the research. Do not answer the question. Just say what the research says w/o any opinions added.
                At the end mention what websites/sources were used (and copy links verbatim if possible)

                The research is:
                {research}
                """
            )
            summary = await model.invoke(prompt)
            logger.info(f"Summary for URL {question.page_url}: {summary}")
            return summary
        except Exception as e:
            if self.use_research_summary_to_forecast:
                raise e  # If the summary is needed for research, then returning the normal error message as the research will confuse the AI
            logger.warning(f"Could not summarize research. {e}")
            return f"{e.__class__.__name__} exception while summarizing research"

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant is providing you with the following research report:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        model = self._next_model()
        reasoning = await self.get_llm(model, "llm").invoke(prompt)
        logger.info(f"Reasoning by {model} for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant is providing you with the following research report:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        model = self._next_model()
        reasoning = await self.get_llm(model, "llm").invoke(prompt)
        logger.info(f"Reasoning by {model} for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        model = self._next_model()
        reasoning = await self.get_llm(model, "llm").invoke(prompt)
        logger.info(f"Reasoning by {model} for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message


        

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = FallTemplateBot2025(
        research_reports_per_question=1,
        # predictions_per_research_report=5,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        enable_summarize_research=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
               model="openrouter/google/gemini-2.5-pro-preview-03-25",
               temperature=0.3,
               timeout=40,
               allowed_tries=2,
             ),
            "model0": GeneralLlm(
               # model="openrouter/google/gemini-2.5-pro-preview-03-25",
               model="openrouter/moonshotai/kimi-k2:free",
               temperature=0.3,
               timeout=40,
               allowed_tries=2,
             ),
            "model1": GeneralLlm(
               # model="openrouter/anthropic/claude-3.7-sonnet",
               model="openrouter/deepseek/deepseek-r1-0528:free",
               temperature=0.3,
               timeout=40,
               allowed_tries=2,
             ),
            "model2": GeneralLlm(
               # model="openrouter/deepseek/deepseek-chat-v3-0324",
               model="openrouter/deepseek/deepseek-chat-v3-0324:free",
               temperature=0.3,
               timeout=40,
               allowed_tries=2,
             ),            
            "model3": GeneralLlm(
               # model="openrouter/x-ai/grok-3-beta",
               model="openrouter/tngtech/deepseek-r1t2-chimera:free",
               temperature=0.3,
               timeout=40,
               allowed_tries=2,
             ),
            "model4": GeneralLlm(
               # model="openrouter/openai/gpt-4.1",
               model="openrouter/qwen/qwen3-235b-a22b:free",
               temperature=0.3,
               timeout=40,
               allowed_tries=2,
             ),
            "model5": GeneralLlm(
               # model="openrouter/google/gemini-2.5-flash-preview",
               model="openrouter/google/gemini-2.0-flash-exp:free",
               temperature=0.3,
               timeout=40,
               allowed_tries=2,
             ),
            # "summarizer": "openai/gpt-4o-mini",
            # "summarizer": "openrouter/moonshotai/kimi-k2:free",
            "summarizer": "openrouter/qwen/qwen3-235b-a22b:free",
            # "researcher": "asknews/deep-research/medium-depth",
            # "researcher": "openrouter/perplexity/sonar-reasoning",
            "researcher": "asknews/news-summaries",
            # "researcher": "openrouter/moonshotai/kimi-k2:free",
            # "researcher": "openrouter/deepseek/deepseek-r1-0528:free",
            # "parser": "openai/gpt-4o-mini",
            # "parser": "openrouter/google/gemini-2.0-flash-exp:free",
            # "parser": "openrouter/moonshotai/kimi-k2:free",
            "parser": "openrouter/qwen/qwen3-235b-a22b:free",
        },
    )

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)
