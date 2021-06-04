import Events as Events
from utils import discriminator
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger

def bayes_optimize(bound,path,num_random_points,
                   num_iterations,kappa, kappa_decay_delay,
                   kappa_decay):
    optimizer = BayesianOptimization(f=discriminator,
                                     pbounds={'degrees1': (0, bound),
                                              'degrees2': (0, bound),
                                              'degrees3': (0, bound),
                                              'translation1': (0, bound),
                                              'translation2': (0, bound),
                                              'translation3': (0, bound),
                                              }, random_state=2018,
                                     verbose=2)

    logger = JSONLogger(path=path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    initialize_grid(optimizer)
    optimizer.maximize(init_points=num_random_points, n_iter=num_iterations,
                       kappa=kappa, kappa_decay_delay=kappa_decay_delay,
                       kappa_decay=kappa_decay)

def initialize_grid(optimizer, points):
    for point in points:
        optimizer.probe(params={"degrees1": point, "degrees2": point,
                                "degrees3": point,"translation1": point,
                                "translation2": point, "translation3": point},
                        lazy=True,)