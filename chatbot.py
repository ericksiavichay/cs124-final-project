# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens
import re
import numpy as np
import math

from PorterStemmer import PorterStemmer

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'R.O.B.' # responsive, omniscient bot

        self.creative = creative

        self.punctationMarks = list(".:;!?")
        self.movie_db = "./data/movies.txt"

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = movielens.ratings()  
        self.porterStemmer = PorterStemmer()

        # stem all the words in the lexicon
        self.sentiment = {}
        for word in movielens.sentiment().items():
          stemmed = self.porterStemmer.stem(word[0])
          self.sentiment[stemmed] = word[1]
        

        # Binarize the movie ratings before storing the binarized matrix.
        ratings = self.binarize(ratings)
        self.ratings = ratings

        self.num_movies_rated = 0
        self.given_ratings = np.zeros(9125, dtype=int)

        self.recommendations = []
        self.requested_rec = False

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        #############################################################################
        # TODO: Write a short greeting message                                      #
        #############################################################################

        greeting_message = "I would like to know more about yourself or what you like!"

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return greeting_message

    def goodbye(self):
        """Return a message that the chatbot uses to bid farewell to the user."""
        #############################################################################
        # TODO: Write a short farewell message                                      #
        #############################################################################

        goodbye_message = "Thanks for helping me learn! Goodbye"

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return goodbye_message

    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        #############################################################################
        # TODO: Implement the extraction and transformation in this method,         #
        # possibly calling other functions. Although modular code is not graded,    #
        # it is highly recommended.                                                 #
        #############################################################################
        if self.creative:
            response = "I processed {} in creative mode!!".format(line)

        prepocessed_line = self.preprocess(line)
        if self.requested_rec:
          want_req = False
          if "yes" in prepocessed_line.lower():
            want_req = True
          if want_req:
            if len(self.recommendations) == 0:
              self.recommendations = self.recommend(self.given_ratings, self.ratings, 10, self.creative)
            rec = self.recommendations.pop(0)
            self.given_ratings[rec] = -3 # Fill rating so that we do not re-recommend the movie to the user
            response = """You should also check out "{}"! 
              Would you like more recommendations?""".format(self.titles[rec][0])
            self.requested_rec = True
          else:
            self.recommendations.clear()
            self.requested_rec = False
            response = "No worries. How about you tell me about more movies (or type :quit to end session)."
        else:
          movies = self.extract_titles(prepocessed_line)
          if len(movies) <= 0:
            response = """Hm, I can't seem to find a movie in your response. 
              Make sure the title of the movie is in quotation marks."""
          elif len(movies) > 1:
            response = "Whoa, slow down there! Please only tell me about only one movie at a time."
          else:
            movie = movies[0]
            matching_movies = self.find_movies_by_title(movie)
            if len(matching_movies) <= 0:
              response = """I've actually never heard of "{}". 
                Why don't you tell me about another movie.""".format(movie)
            elif len(matching_movies) > 1:
              response = """I found more than one movie called "{}". Can you clarify please?""".format(movie)
            else:
              movie_index = matching_movies[0]
              sentiment = self.extract_sentiment(prepocessed_line)
              if sentiment == None: # Worst case scenario: we do not understand the input and therefore cannot get sentiment
                response = """I'm sorry. I'm really not understanding what you are trying to say. 
                  Please try again, making sure to put the movie title in quotation marks and
                    expressing how you liked it."""
              elif sentiment == 0:
                response = """Hm, I'm not sure if you liked "{}" or not. 
                  Please be a little more specific.""".format(movie)
              elif sentiment > 0:
                self.num_movies_rated += 1
                self.given_ratings[movie_index] = sentiment
                response = """I'm glad you liked "{}"! Tell me about another movie!""".format(movie)
              elif sentiment < 0:
                self.num_movies_rated += 1
                self.given_ratings[movie_index] = sentiment
                response = """I'm sorry to hear that you didn't like "{}". 
                  Let's hear about another movie""".format(movie)
              if self.num_movies_rated >= 5:
                self.recommendations = self.recommend(self.given_ratings, self.ratings, 10, self.creative)
                rec = self.recommendations.pop(0)
                self.given_ratings[rec] = -3 # Fill rating so that we do not re-recommend the movie to the user
                response += """ Given everything you've told me, I think you would like "{}"! 
                  Would you like more recommendations?""".format(self.titles[rec][0])
                self.requested_rec = True

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information from a line of text.

        Given an input line of text, this method should do any general pre-processing and return the
        pre-processed string. The outputs of this method will be used as inputs (instead of the original
        raw text) for the extract_titles, extract_sentiment, and extract_sentiment_for_movies methods.

        Note that this method is intentially made static, as you shouldn't need to use any
        attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        #############################################################################
        # TODO: Preprocess the text into a desired format.                          #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to your    #
        # implementation to do any generic preprocessing, feel free to leave this   #
        # method unmodified.                                                        #
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return text

    def get_titles_between_quotes(self, text):
        """
          Expects text to be preprocessed. We expect that there will be an even
          number of quote characters ("), including zero quote characters.
          Given preprocessed text, we will expect the following:
              Starter mode:
                  Text between quotes must be bounded by the quotes, i.e, we cannot
                  have a string such as " movie 1 ", but instead we must have 
                  "movie 1". This makes it easier to expect where the quotes are,
                  so preprocess function must be developed.
          :param text: preprocessed text
          :returns: empty list if no movies found; list of strings of movies if movies found
        """
          
        movie_titles = []
        start_index = text.find("\"")
        while (start_index != -1):
          next_index = text.find("\"", start_index + 1)
          if (next_index == -1): break
          sub_str = text[start_index + 1:next_index]
          sub_str = sub_str.strip()
          movie_titles.append(sub_str)
          start_index = text.find("\"", next_index + 1)
        return movie_titles 

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially in the text.
        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.
        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess('I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]
        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """

        movie_titles = self.get_titles_between_quotes(preprocessed_input)
        return movie_titles

    def get_year_index(self, title):
        """
        Given a title, checks if it contains a year (\d{4}). Returns index of the occurence.
        
        :param title: string, title of movie
        :returns: (length - 1) if it doesn't have year, or index of the first parenthesis of the year in the string
        """
        year_pattern = "\(\d{4}\)" # with this specific patter, descriptions within parenthesis will be ignored
        if not re.search(year_pattern, title):
            return len(title)
        for match in re.finditer(year_pattern, title):
            return match.start(0)

    def break_by_article(self, title):
        broke = ["", title]
        articles = ["The", "A", "An"]
        for art in articles:
            if art in title:
                remaining = re.split(art, title)[1]
                return [art, remaining.strip()]
        return broke

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.
        - If no movies are found that match the given title, return an empty list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a list
        that contains the index of that matching movie.
        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 1953]
        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        matching_movie_indices = []
        year_index = self.get_year_index(title)
        yearless_title, year = title[:year_index], title[year_index:] # year will be empty string if it doesn't contain year
        yearless_title = yearless_title.strip()
        year = " " + year
        article, remaining = self.break_by_article(yearless_title) 
        if (article != ""):
            rearranged = remaining + ", " + article + year
        else: rearranged = remaining + year
        rearranged = rearranged.strip()

        # given this rearranged format, search for this in the dataset
        # a title with year should return list of length one since 
        # it would not be a substring of anything else but the exact match
        with open(self.movie_db) as movie_file:
            lines = movie_file.readlines()
            for line in lines:
                movie_index, movie_title, _ = line.split("%") # splitting should return 3 strings
                if movie_title == rearranged: return [int(movie_index)]
                if (rearranged + " (") in movie_title:
                    matching_movie_indices.append(int(movie_index))
        
        return matching_movie_indices

    
    def processInputForSentimentExtraction(self, preprocessed_input):
        # add a space before after each punctation mark
        ch = 0
        while(True):
          if (ch >= len(preprocessed_input)): break
          # if this char is a punctation mark
          if preprocessed_input[ch] in self.punctationMarks:
            preprocessed_input = preprocessed_input[:ch] + " " + preprocessed_input[ch:]
            ch += 1
          ch += 1

        #print("...After adding space before all punctation: " + preprocessed_input)

        # remove title
        while (True):
          startQuote = preprocessed_input.find('\"')
          if (startQuote == -1): 
            break
          endQuote = preprocessed_input.find('\"', startQuote + 1)
          if (endQuote == -1): 
            print("ERROR")
            break
          beg = preprocessed_input[:startQuote]
          end = preprocessed_input[endQuote + 1:]
          preprocessed_input = beg + end
        # print("...After removing title: " + preprocessed_input)

        # stem the words
        stemmedInput = preprocessed_input.split(" ")
        for i, word in enumerate(stemmedInput):
          # eleminate empty strings in list
          if (word == ""): 
            stemmedInput.pop(i)
            continue
          # otherwise stem the word
          stemmedInput[i] = self.porterStemmer.stem(word)
        #print("...After stemming: ", stemmedInput)
        
        # account for negatations
        pattern = re.compile("(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't")
        for i, word in enumerate(stemmedInput):
          # if word is a negation
          # print("MATCHING PATTERN: ", pattern)
          # print(word)
          if pattern.search(word) != None:
            # print("YIKES: Negation found")
            # change every word until a punctation to NOT_
            if (i == len(stemmedInput) - 1): break # ensure this isn't the last word in the list
            j = i + 1
            while j < len(stemmedInput):
              toNegate = stemmedInput[j]
              # stop at punctation
              if (re.search(r"^[.:;!?]$", toNegate)): break
              # otherwise manipulate
              toNegate += "_NEG"
              stemmedInput[j] = toNegate
              j += 1

        #print("...After negations: ", stemmedInput)

        return stemmedInput
    

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the text
        is super negative and +2 if the sentiment of the text is super positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess('I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        # to test: I never liked "Titanic (1997)".

        # remove word between quotation marks
        processed_input = self.processInputForSentimentExtraction(preprocessed_input)
        # processed_input is a list of all the tokens in this sentence, not including the title

        # extract sentiment
        lambda_val = 1
        # extract sentiment 
        pos_count = 1
        neg_count = 1
        for word in processed_input:
          # check if word is negated
          word_is_neg = False
          if (word[-4:] == "_NEG"):
            word = word[:-4]
            word_is_neg = True

          # ensure the word is in our lexicon
          if (word in self.sentiment):
            if (word == "terrible"):
              print("------Found the world terrible")
              
            val = self.sentiment[word]
            # accomodate negative words
            if (val == "pos"):
              if (word_is_neg): 
                neg_count += 1
              else: 
                pos_count += 1
            else:
              if (word_is_neg):
                pos_count += 1
              else: 
                neg_count += 1

        if ((pos_count / neg_count) > lambda_val):
          sentiment = 1
        elif ((neg_count / pos_count) > lambda_val):
          sentiment = -1
        else:
          sentiment = 0
        return sentiment

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of pre-processed text
        that may contain multiple movies. Note that the sentiments toward
        the movies may be different.

        You should use the same sentiment values as extract_sentiment, described above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess('I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie title,
          and the second is the sentiment in the text toward that movie
        """
        movies_sentiment = list()
        # break input into CLAUSES by punctation marks or but/however/and
        print("inside extract sentiment for movies function")
        return

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least edit distance
        from the provided title, and with edit distance at most max_distance.

        - If no movies have titles within max_distance of the provided title, return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given title
          than all other movies, return a 1-element list containing its index.
        - If there is a tie for closest movie, return a list with the indices of all movies
          tying for minimum edit distance to the given movie.

        Example:
          chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
        """

        pass

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be talking about
        (represented as indices), and a string given by the user as clarification
        (eg. in response to your bot saying "Which movie did you mean: Titanic (1953)
        or Titanic (1997)?"), use the clarification to narrow down the list and return
        a smaller list of candidates (hopefully just 1!)

        - If the clarification uniquely identifies one of the movies, this should return a 1-element
        list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it should return a list
        with the indices it could be referring to (to continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by the clarification
        """
        pass

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use any
        attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered positive

        :returns: a binarized version of the movie-rating matrix
        """

        for i in range(len(ratings)):
          for j in range(len(ratings[0])):
            rating = ratings[i][j]
            if rating == 0:
              continue
            if rating > threshold:
              ratings[i][j] = 1
            else:
              ratings[i][j] = -1

        return ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        #############################################################################
        # TODO: Compute cosine similarity between the two vectors.
        #############################################################################

        numerator = 0
        denom_u = 0
        denom_v = 0
        for i in range(len(u)):
          rank_u = u[i]
          rank_v = v[i]
          if rank_u != 0 and rank_v != 0:
            numerator += rank_u * rank_v
            denom_u += pow(rank_u, 2)
            denom_v += pow(rank_v, 2)

        denominator = math.sqrt(denom_u) * math.sqrt(denom_v)
        if denominator == 0:
          similarity = 0
        else:
          similarity = numerator / denominator

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in ratings_matrix,
          in descending order of recommendation
        """

        #######################################################################################
        # TODO: Implement a recommendation function that takes a vector user_ratings          #
        # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
        # Do not use the self.ratings matrix directly in this function.                       #
        #                                                                                     #
        # For starter mode, you should use item-item collaborative filtering                  #
        # with cosine similarity, no mean-centering, and no normalization of scores.          #
        #######################################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        already_rated = []

        # Find all movies already rated by user
        for i in range(len(user_ratings)):
          if user_ratings[i] != 0:
            already_rated.append(i)
        
        # Get estimated ratings for movies not rated by user
        all_rated = []
        for i in range(len(user_ratings)):
          if user_ratings[i] != 0:
            # If already rated, append -3 to ensure it will not be considered when giving recommendation
            all_rated.append(-3)
          else:
            # Compute nominator and denominator in weighted average formula
            nominator = 0
            denominator = 0
            for rated_movie_index in already_rated:
              # Compute cosine similarity between movie i and movie rated by user with index rated_movie_index
              rated_movie_vector = ratings_matrix[rated_movie_index].copy()
              movie_to_rate_vector = ratings_matrix[i].copy()
              sim = self.similarity(rated_movie_vector, movie_to_rate_vector)
              nominator += sim * user_ratings[rated_movie_index]
              denominator += sim

            if denominator == 0:
              ranking = 0
            else:
              ranking = nominator / denominator
            all_rated.append(ranking)
            
        # Get top k recommendations
        sorted_ranks = []
        for rank in sorted(all_rated, reverse=True):
          if rank != 3:
            sorted_ranks.append(rank)
        for i in range(min(k, len(sorted_ranks))):
          movie_index = all_rated.index(sorted_ranks[i])
          all_rated[movie_index] = -3
          recommendations.append(movie_index)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return recommendations

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
        """Return debug information as a string for the line string from the REPL"""
        # Pass the debug information that you may think is important for your
        # evaluators
        debug_info = 'debug info'
        return debug_info

    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your chatbot
        can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6 instructions.
        Remember: in the starter mode, movie names will come in quotation marks and
        expressions of sentiment will be simple!
        Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, run:')
    print('    python3 repl.py')
