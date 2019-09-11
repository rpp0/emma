import redis
import redis_lock


class MemStore():
    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.db = redis.StrictRedis.from_url(redis_url)

    def set(self, key, value):
        self.db.set(key, value)

    def get(self, key):
        return self.db.get(key)

    def delete(self, key):
        self.db.delete(key)

    def reset_lock(self, key):
        lock = redis_lock.Lock(self.db, key)
        lock.reset()
