def build_feather_timelines():
    timelines = []
    UserHistory = namedtuple("UserHistory", "history books ratings")

    df = pd.read_feather('ratings.feather')
    for _, group in df.groupby('user_id'):
        books = group['work_id'].tolist()
        ratings = group['rating'].tolist()
        size = len(books)
        timelines.append(UserHistory(size, books, ratings))
    return timelines


def create_contexts(user, max_length=10):
    contexts = []
    for end in range(1, len(user.books)):
        current = max(0, end - max_length)
        label_id = [user.books[end]]
        context_id = user.books[current:end]
        context_rating = user.ratings[current:end]
        while len(context_id) < max_length:
            context_id.append(0)
            context_rating.append(0)
        contexts.append([label_id, context_id, context_rating])
    return contexts


def timelines_to_contexts(max_length=10, training_split=0.9):
    records = []
    timeline = build_feather_timelines()
    logger.info("Timelines built")
    for user in timeline:
        user_examples = create_contexts(user, max_length)
        records.extend(user_examples)
    return pd.DataFrame(records, columns=['label_id', 'context_id', 'context_rating'])