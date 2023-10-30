from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    PromptTemplate,
)

from nltk.tokenize import sent_tokenize
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

frq_quality_rubric = ["Can the question be answered using only evidence from the context above?",
                      "Does the question test understanding of the information presented in the context?"]
feedback_quality_rubric = ["contradict itself",
                           "repeat the same points"]

class TeachingStaff:
    def __init__(self, api_key, standard, topic=None, rubric=None, enable_qa=False):
        chatllm = ChatOpenAI(openai_api_key=api_key, model_name='gpt-4')  # model_name='gpt-4'
        self.llm = OpenAI(openai_api_key=api_key, model_name='gpt-4')  # model_name='gpt-4'

        if topic != None:
            self.set_topic(topic)
        self.standard = standard

        self.knowledge_gen_chain, self.question_asking_chain, \
            self.t_chain, self.s_chain, self.quality_chain = self.initialize_agents(chatllm)

        if rubric != None:
            self.rubric = rubric
        else:
            self.rubric = self.generate_rubric()

        self.enabled_qa = enable_qa

    def set_topic(self, topic):
        self.topic = topic

    def get_rubric(self):
        return self.rubric

    def initialize_agents(self, llm):
        # knowledge generation chain
        kg_template = """Mary is an super-intelligent, advanced AI task executor that posseses accurate knowledge on every topic.
       It does not mention itself or admit its nature as an AI.
       It uses the voice of a primary or secondary source.
       It does not use first or second person except when quoting a source.
       It fulfills requests exactly and concisely
       For the following requests, you will respond and do tasks as Mary.
  
       {chat_history}
       Human: {task}
       Mary:"""
        kg_prompt = PromptTemplate(input_variables=["chat_history", "task"], template=kg_template)
        kg_memory = ConversationBufferMemory(memory_key="chat_history")
        kg_chain = LLMChain(llm=llm, prompt=kg_prompt, verbose=False, memory=kg_memory)

        # question asking chain
        qa_template = f"""John is an super-intelligent question asking AI with critical reading and thinking skills.
     It does not mention itself or admit its nature as an AI.
     It does not use first or second person except when quoting a source.
     It specializes in generating insightful questions that test understanding of a text passage.
     Its questions expect the response to exhibit the standard: {self.standard}.
     For the following requests, you will respond and do tasks as John.\n\n""" + \
                      """
     {chat_history}
     Human: {task}
     John:"""
        qa_prompt = PromptTemplate(input_variables=["chat_history", "task"], template=qa_template)
        qa_memory = ConversationBufferMemory(memory_key="chat_history")
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt, verbose=False, memory=qa_memory)

        # teaching chain
        t_template = """Susan is an super-intelligent, advanced educational AI that is an expert at teaching students.
     It is knowledgable in all common student mistakes.
     It does not mention itself or admit its nature as an AI.
     It does not use first or second person except when quoting a source.
     It fulfills requests exactly and concisely
     It is extremely competent in evaluating free response questions, identifying false information,
     and providing the best advice for students to improve their writing with respect to a rubric.
     For the following requests, you will respond and do tasks as Susan.

     {chat_history}
     Human: {task}
     Susan:"""
        t_prompt = PromptTemplate(input_variables=["chat_history", "task"], template=t_template)
        t_memory = ConversationBufferMemory(memory_key="chat_history")
        t_chain = LLMChain(llm=llm, prompt=t_prompt, verbose=False, memory=t_memory)

        # student chain (simulate student for testing)
        s_template = """Zach is a human 4th grade student that is doing a writing assignment.
    For the following requests, you will respond and write as Zach.

    {chat_history}
    {task}
    Zach:"""
        s_prompt = PromptTemplate(input_variables=["chat_history", "task"], template=s_template)
        s_memory = ConversationBufferMemory(memory_key="chat_history")
        s_chain = LLMChain(llm=llm, prompt=s_prompt, verbose=False, memory=s_memory)

        # quality assurance agent
        quality_template = """Jan is an advanced teaching quality assurance AI that is an expert at
    editing educational text to promote concise, helpful, and easy to understand material for students.
    It does not mention itself or admit its nature as an AI.
    It fulfills requests exactly and concisely without repeating the request.
    It does not use first or second person except when quoting a source.
    For the folling requests, you will respond as Jan.

    {chat_history}
    {task}
    Jan:"""
        quality_prompt = PromptTemplate(input_variables=["chat_history", "task"], template=quality_template)
        quality_memory = ConversationBufferMemory(memory_key="chat_history")
        quality_chain = LLMChain(llm=llm, prompt=quality_prompt, verbose=False, memory=quality_memory)

        return kg_chain, qa_chain, t_chain, s_chain, quality_chain

    def generate_FRQ(self, max_retry=2):
        intro, context = self.generate_intro_context()
        question = self.generate_question(context)
        passed_qa = False
        tries = 0
        while not passed_qa and self.enabled_qa and tries <= max_retry:
            passed_qa, failed_check = self.check_FRQ_quality(intro + "\n" + context, question)
            if not passed_qa:
                print("discarded question: " + question + "\n due to failing to pass: " + failed_check + "\n")
                question = self.generate_question(context)
                tries = tries + 1

        return intro + "\n" + context, question

    def check_FRQ_quality(self, context, question):
        prompt = f"""Refer to the following context and question for the following requests:
    <CONTEXT>
    {context}
    </CONTEXT>
    <QUESTION>
    {question}
    </QUESTION>
    """
        self.quality_chain.predict(task=prompt)

        for check in frq_quality_rubric:
            if ("YES" not in self.quality_chain.predict(task=check + " Answer with one word YES or NO.")):
                return False, check
        return True, ""

    def check_feedback_quality(self, feedback):
        prompt = f"""Refer to the following feedback for the following requests:
    <FEEDBACK>
    {feedback}
    </FEEDBACK>
    """
        self.quality_chain.predict(task=prompt)

        for check in feedback_quality_rubric:
            if ("YES" in self.quality_chain.predict(
                    task="Does the feedback " + check + "? Answer with one word YES or NO.")):
                return False, check
        return True, ""

    def generate_intro_context(self):
        intro_prompt = f"Write a short 2 sentence introduction for the topic '{self.topic}'"
        intro = self.knowledge_gen_chain.predict(task=intro_prompt)
        knowledge_prompt = f"generate 8 facts related to the topic '{self.topic}' which could follow your introduction"
        knowledge = self.knowledge_gen_chain.predict(task=knowledge_prompt)
        context_prompt = f"generate 3 paragraphs about the topic naturally utilizing the above facts incorporating quotes if necessary"
        context = self.knowledge_gen_chain.predict(task=context_prompt)
        return intro, context

    def generate_question(self, context):
        prompt = f"Generate an open-ended question which can be answered solely by drawing evidence from the context:\n'{context}'"
        question = self.question_asking_chain.predict(task=prompt)
        return question

    def generate_rubric(self):
        prompt = \
            f"""Concisely generate a rubric for grading an essay answer based on how well it demonstrates the core standard '{self.standard}'.
    It should assign a score from 1-3 and give criteria for meeting each score cutoff. """
        return self.t_chain.predict(task=prompt)

    def generate_response(self, frq):
        prompt = f"Answer the following question in paragraph form:\n {frq}"
        return self.s_chain.predict(task=prompt)

    def try_n_times(self, prompt, condition, n):
        for i in range(0, n):
            if condition(self.llm(prompt)):
                return True
        return False

    def evaluate_correctness(self, context, q, student_response):
        sentences = sent_tokenize(student_response.strip())
        correct = True
        incorrect_sentences = []
        for sentence in sentences:
            correctness = f"""Identify claims from the response and and evaluate accuracy of each using evidence from the context step by step. Evidence from the context is not needed if the claim is common sense. Return a final answer of CORRECT if all claims are accurate, and INCORRECT otherwise..
      Example1:
      <CONTEXT>
      </CONTEXT>
      <RESPONSE>
      During the Scramble for Africa, European powers justified their colonization efforts by claiming to bring civilization, Islam, and economic development to the continent.
      </RESPONSE>
      Evaluation:
      Claim 1: European powers justified their colonization efforts by bringing civilization to the continent. \n\nAccuracy: CORRECT. The text states that European powers "justified their colonization efforts by claiming to bring civilization, Christianity, and economic development to the continent." \n\nClaim 2: European powers justified their colonization efforts by bringing Islam to the continent. \n\nAccuracy: INCORRECT. The text states that European powers "justified their colonization efforts by claiming to bring civilization, Christianity, and economic development to the continent." Islam is not mentioned as one of the claims made by European powers.
      Final Answer: INCORRECT

      <CONTEXT>
      {context}
      </CONTEXT>
      <RESPONSE>
      {sentence}
      </RESPONSE>
      Evaluation:
      """
            evaluation = self.try_n_times(correctness, lambda x: x.split("\n")[-1].split(" ")[-1].strip() == "CORRECT",
                                          3)
            if not evaluation:
                incorrect_sentences.append(sentence)
            correct = correct and evaluation
        prompt = f"""Does the response answer the question? Remember that a response with incorrect information can still answer the question if the misinformation does not overly impact the main points of the response. Return one word YES or NO:
    Example1:
    <QUESTION>
    </QUESTION>
    <RESPONSE>
    </RESPONSE>
    Answer: YES

    <QUESTION>
    {q}
    </QUESTION>
    <RESPONSE>
    {student_response}
    </RESPONSE>
    """
        yesno = self.try_n_times(prompt, lambda x: x.split("\n")[-1].split(" ")[-1].strip() == "YES", 2)
        return (correct, incorrect_sentences, yesno)

    def evaluate_core_standard(self, context, student_response):
        prompt = f"""
    {self.rubric}
    Score the following response from 1-3. Remember the student can only
    use information provided in the context and scale the score accordingly. Explain why step by step.
    Then, unless the score is 3, give suggestions on how to improve the score.
    <CONTEXT>
    {context}
    </CONTEXT>
    <RESPONSE>
    {student_response}
    </RESPONSE>
    """
        feedback = self.t_chain.predict(task=prompt)
        return feedback

    def evaluate_response(self, context, question, student_response, max_edits=2):
        correct, incorrect_sentences, answered_question = self.evaluate_correctness(context, question, student_response)
        standard_feedback = self.evaluate_core_standard(context, student_response)
        if correct and answered_question:
            feedback = "Good Job! You answered the question correctly.\n"
        elif correct and not answered_question:
            feedback = "Your response was accurate, but did not adequately answer the question.\n"
            prompt = f"Explain step by step why the response does not adequately answer the question: {question}"
            feedback += self.t_chain.predict(task=prompt) + "\n"
        elif not correct and answered_question:
            feedback = "Your response contained incorrect information but overall still answered the question.\n"
            for sentence in incorrect_sentences:
                prompt = f"Explain step by step why the sentence '{sentence}' is incorrect given the provided context"
                feedback += self.t_chain.predict(task=prompt) + "\n"
        else:
            feedback = "Your response contained incorrect information and did not adequately answer the question.\n"
            for sentence in incorrect_sentences:
                prompt = f"Explain step by step why the sentence '{sentence}' is incorrect given the provided context"
                feedback += self.t_chain.predict(task=prompt) + "\n"
            prompt = f"Explain step by step why the response does not adequately answer the question: {question}"
            feedback += self.t_chain.predict(task=prompt) + "\n"
        feedback = feedback + "\n\n" + standard_feedback

        passed_qa = False
        num_edits = 0
        while not passed_qa and num_edits <= max_edits and self.enabled_qa:
            passed_qa, failed_check = self.check_feedback_quality(feedback)
            if not passed_qa:
                prompt = f"""Return an edited version of this feedback so that it does not {failed_check}.
        <FEEDBACK>
        {feedback}
        </FEEDBACK>
        """
                print(f"Improving feedback so that it does not {failed_check}")
                num_edits += 1
                feedback = self.quality_chain.predict(task=prompt)
        return feedback


if __name__ == "__main__":
    OPENAI_API_KEY = "sk-----------"
    TA = TeachingStaff(OPENAI_API_KEY,
                       standard="Draw evidence from literary or informational texts to support analysis, reflection, and research.",
                       enable_qa=True)
    TA.set_topic("Dinosaur extinction event")
