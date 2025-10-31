import copy
import unittest

from data_loader import load_mnist
from fl_base import evaluate, fedavg_aggregate, train_local
from ga_selection import ga_client_selection
from model import create_model


class TestFL(unittest.TestCase):
    def setUp(self):
        self.num_clients = 5
        self.client_datasets, self.test_dataset = load_mnist(self.num_clients, 0.5)
        self.global_model = create_model()

    def test_data_loading(self):
        self.assertEqual(len(self.client_datasets), self.num_clients)
        self.assertGreater(len(self.test_dataset), 0)

    def test_model_creation(self):
        model = create_model()
        self.assertIsNotNone(model)

    def test_evaluation(self):
        acc, loss = evaluate(self.global_model, self.test_dataset)
        self.assertIsInstance(acc, float)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    def test_train_local(self):
        trained = train_local(copy.deepcopy(self.global_model), self.client_datasets[0])
        self.assertIsNotNone(trained)

    def test_fedavg_aggregate(self):
        local_models = [
            train_local(copy.deepcopy(self.global_model), self.client_datasets[i])
            for i in range(3)
        ]
        aggregated = fedavg_aggregate(
            local_models, [self.client_datasets[i] for i in range(3)],
        )
        self.assertIsNotNone(aggregated)

    def test_ga_selection(self):
        local_accs = [0.5 + 0.1 * i for i in range(self.num_clients)]  # mock
        selected = ga_client_selection(
            self.num_clients, k=3, pop_size=10, generations=2, local_accs=local_accs,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)  # unique


if __name__ == "__main__":
    unittest.main()
