import os


class UnitTestSettings:
    TEST_FAST = os.environ.get('EMMA_FAST_UNITTEST', 'False') == 'True'
