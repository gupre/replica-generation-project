from clearml import Task
import random
import time

# создаём задачу в ClearML
task = Task.init(
    project_name="Test Project",
    task_name="Connection Test"
)

# логгер
logger = task.get_logger()

# логируем несколько метрик
for i in range(5):
    value = random.random()
    logger.report_scalar(
        title="random_metric",
        series="test_series",
        value=value,
        iteration=i
    )
    time.sleep(1)

print("Done! Check ClearML Web UI.")
