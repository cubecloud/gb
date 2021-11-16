import sys
import time
import pytz
import copy
import random
import pandas as pd
import operator
from functools import reduce
import datetime
from datetimerange import DateTimeRange
from itertools import cycle
from itertools import combinations

import numpy as np
__version__ = 0.005

_FLOAT_PRECISION_ = 1


def pos_cycle(my_list, start_at=None):
    start_at = 0 if start_at is None else my_list.index(start_at)
    while True:
        yield my_list[start_at]
        start_at = (start_at + 1) % len(my_list)
    pass


def datetime_now():
    try:
        timezone
    except NameError:
        return datetime.datetime.now()
    else:
        return datetime.datetime.now(timezone)


def get_combo_of_iter_params(params: list or tuple) -> tuple:
    length = len(params)
    # iterative_type = 'num'
    typeof = None
    if length == 1 and isinstance(params, list):
        iterative_type = 'number'
        typeof = (type(params), type(params[0]), f'{iterative_type}')
    elif length >= 2 and isinstance(params, list):
        iterative_type = 'sequence'
        typeof = (type(params), type(params[0]), f'{iterative_type}')
    elif length == 2 and isinstance(params, tuple):
        iterative_type = 'ceil'
        typeof = (type(params), type(params[0]), f'{iterative_type}')
    elif length == 2 and isinstance(params, set):
        iterative_type = 'range'
        params = list(params)
        typeof = (type(params), type(params[0]), f'{iterative_type}')
    return typeof


def no_func(param):
    return param


def get_param_func_and_args(typeof, gene_name, gene_params):
    if typeof is None:
        msg = f'Unknown gene type {gene_name} {type(gene_params)}'
        sys.exit(msg)

    gene_dict = {(tuple, int, 'ceil'): random.randint,
                 (tuple, float, 'ceil'): random.uniform,
                 (list, int, 'number'): no_func,
                 (list, float, 'number'): no_func,
                 (set, int, 'range'): range,
                 (list, int, 'sequence'): random.sample,
                 (list, str, 'sequence'): random.sample,
                 (list, float, 'sequence'): random.sample,
                 (list, object, 'sequence'): random.sample,
                 }
    func = gene_dict.get(typeof, None)
    args = []
    if isinstance(gene_params, tuple):
        args = [gene_params[0], gene_params[1]]
    elif isinstance(gene_params, list):
        if typeof[2] == 'sequence':
            args = [gene_params, 1]
        elif typeof[2] == 'number':
            args = [gene_params]
    elif isinstance(gene_params, set):
        gene_params = list(gene_params)
        args = [gene_params[0], gene_params[1]]
    else:
        msg = f'Unknown types combination {gene_name} {type(gene_params)}'
        sys.exit(msg)
    return func, args


def count_params(gene_params):
    """

    Args:
        gene_params (tuple or list):    gene parameters

    Returns:
        gene_iter (int):                calculated gene iterations
        items:                          gene molecules (composition items)
    """
    params_types_func = {(tuple, int, 'ceil'): range,
                         (tuple, float, 'ceil'): np.arange,
                         (list, int, 'number'): len,
                         (list, float, 'number'): len,
                         (set, int, 'range'): range,
                         (list, int, 'sequence'): len,
                         (list, str, 'sequence'): len,
                         (list, float, 'sequence'): len,
                         (list, object, 'sequence'): len,
                         }
    typeof = get_combo_of_iter_params(gene_params)
    func = params_types_func.get(typeof, None)
    gene_iter: int = 0
    items = []
    if isinstance(gene_params, tuple):
        if typeof[1] is int:
            items = func(gene_params[0], gene_params[1])
            gene_iter = len(items)
        else:
            step = 1 / 10 ** _FLOAT_PRECISION_
            items = func(gene_params[0], gene_params[1], step)
            gene_iter = len(items)
    elif isinstance(gene_params, list):
        if typeof[2] == 'sequence':
            items = gene_params
            gene_iter = func(gene_params)
        elif typeof[2] == 'number':
            items = gene_params[0]
            gene_iter = 1
    elif isinstance(gene_params, set):
        gene_params = list(gene_params)
        items = func(gene_params[0], gene_params[1])
        gene_iter = len(items)
    else:
        msg = f'Unknown types combination {gene_params} {type(gene_params)}'
        sys.exit(msg)
    return gene_iter, items


def get_iter_count(genes, root_gene):
    params_count = []
    for idx, (gene_name, gene_params) in enumerate(genes.items()):
        params_count.append(count_params(gene_params)[0])
        if root_gene:
            key = list(root_gene.keys())[0]
            if key == gene_name:
                params_count[idx] = 1
    iter_number = reduce(operator.mul, params_count, 1)
    return iter_number, params_count


class Gene:
    """ Gene Class """

    def __init__(self, gene_name, gene_params, gene_mutation_probability) -> None:
        """ Create the gene

        Args:
            gene_name (string):                 name of Gene
            gene_params (int, tuple or list):   params for gene generation
            gene_mutation_probability (float):  probability of *this* gene mutation
        Returns:
            None
        """
        self.gene_name = gene_name
        self.gene_params = gene_params
        self.gene_mutation_probability = gene_mutation_probability

        self.typeof = get_combo_of_iter_params(gene_params)
        self.func, self.args = get_param_func_and_args(self.typeof, self.gene_name, self.gene_params)
        self.gene_iter, self.gene_items = count_params(self.gene_params)

        if self.gene_iter == 1:
            self.gene_freeze = True
            self.gene_items_qty = 1
            self.itself = self.gene_params[0]
            self.mutation_positions_move = 1
        else:
            self.gene_freeze = False
            self.gene_items_qty = len(self.gene_items)
            self.mutation_positions_move = int(self.gene_items_qty // 2)
            self.start_mutation = 1
            if self.mutation_positions_move < 2 or self.gene_items_qty == 2:
                self.mutation_positions_move = 2

            self.itself = self.func(*self.args)
            if isinstance(self.itself, list):
                self.itself = self.itself[0]
            self.items_cycler = pos_cycle(self.gene_items, self.gene_items.index(self.itself))
        pass

    def re_generate_gene(self):
        """ re-Generate the gene value. if not self.gene_freeze == True
        Returns
        -------
        None
        """
        if not self.gene_freeze:
            # self.itself = _get_generated_param(self.typeof, self.gene_name, self.gene_params)
            self.itself = self.func(*self.args)
            if isinstance(self.itself, list):
                self.itself = self.itself[0]
        pass

    def mutate_gene(self):
        """ Mutate the gene value with mutation_probability. if not self.gene_freeze == True
        Returns
        -------
        None
        """
        if not self.gene_freeze:
            if random.random() <= self.gene_mutation_probability:
                # randomizing of steps q-ty for move gene molecule (step 1 is current position)
                number = random.randint(self.start_mutation, self.mutation_positions_move)
                idx = self.gene_items.index(self.itself)
                self.items_cycler = pos_cycle(self.gene_items, idx)
                for ix in range(number):
                    self.itself = next(self.items_cycler)
        pass

    def freeze_gene(self):
        """ Freeze the gene from mutation and re-generation
        Returns
        -------
        None
        """
        self.gene_freeze = True
        pass

    def unfreeze_gene(self):
        """ unFreeze the gene for mutation and re-generation
        Returns
        -------
        None
        """
        self.gene_freeze = False
        pass


class Bot:
    bots_history = dict()
    dupe_counter = dict()
    flag_done = dict()
    scoreboard = dict()

    def __init__(self,
                 population_id,
                 genes,
                 root_gene,
                 share_history,
                 store_history,
                 bot_mutation_probability=0.15,
                 gene_mutation_probability=0.10) -> None:
        """
        Create the Bot with genes

        Parameters
        ----------
        population_id: int                  population_id for bots history
        genes: dict                         dictionary with genes_name and gene data to iterate
        share_history: bool                 sharing history between bots flag
        store_history: bool                 store history for save and load flag (decrease productivity)
        bot_mutation_probability: float     (0.0-1.0) probability
        root_gene: dict                     key = name of the gene, value = value of the gene to freeze

        Returns
        -------
        None
        """
        self.population_id = population_id
        self.__class__.bots_history[self.population_id]: set = set()
        self.__class__.dupe_counter[self.population_id]: int = 0
        self.__class__.flag_done[self.population_id]: bool = False
        self.__class__.scoreboard[self.population_id]: dict = {}
        self.bot_genome = dict()
        self.root_gene = root_gene
        self.share_history = share_history
        self.store_history = store_history
        self.bot_mutation_probability = bot_mutation_probability
        self.gene_mutation_probability = gene_mutation_probability
        self.genes = genes
        self.genes_qty = len(self.genes)
        self.score: float = np.NAN
        self.bot_freeze = False
        self.possible_iterations, _ = get_iter_count(self.genes, self.root_gene)
        self._init_bot()
        pass

    def _init_bot(self) -> None:
        """
        Initialize bot genes and freezing root_gene if necessary

        Returns
        -------
        None
        """
        for idx, (gene_name, gene_params) in enumerate(self.genes.items()):
            self.bot_genome.update({idx: Gene(gene_name, gene_params, self.gene_mutation_probability)})
            if self.root_gene:
                key = list(self.root_gene.keys())[0]
                if key == gene_name:
                    self.bot_genome[idx].itself = self.root_gene.get(key)
                    self.bot_genome[idx].freeze_gene()
        pass

    @classmethod
    def set_bots_history(cls, bots_history: set, population_id: int or str) -> None:
        """ Setting bots_history for population with population_id

        Parameters
        ----------
        bots_history: set       set of tuples
        population_id: int      id of this population

        Returns
        ----------
        None
        """
        cls.bots_history[population_id] = bots_history
        pass

    @classmethod
    def get_bots_history(cls, population_id: int or str) -> set:
        """ Get history of population with population_id from class dictionary
        Parameters
        ----------
        population_id : int                       id of this population

        Returns
        ----------
        cls.bots_history[population_id]: set      return set of tuples
        """
        return cls.bots_history[population_id]

    @classmethod
    def set_bots_flag_done(cls, population_id: int or str, flag: bool) -> None:
        """ Set flag_done of population with population_id
        Parameters
        ----------
        population_id : int or str     id of this population
        flag : bool                  bool flag

        """
        cls.flag_done[population_id] = flag
        pass

    @classmethod
    def get_bots_flag_done(cls, population_id: int or str) -> bool:
        """ Set flag_done of population with population_id
        Parameters
        ----------
        population_id : int                       id of this population

        """
        return cls.flag_done[population_id]

    @classmethod
    def get_bots_dupes_counter(cls, population_id: int or str) -> int:
        """ Get dupe_counter num from class
        Parameters
        ----------
        population_id : int or str                       id of this population

        Returns
        ----------
        cls.dupe_counter[population_id]: int      return num of dupes generated until now
        """
        return cls.dupe_counter[population_id]

    def add_bot_genome_to_history(self) -> None:
        """
        Add bot genome data to history

        """
        if self.share_history:
            history = Bot.get_bots_history(self.population_id)
            if self.possible_iterations != len(history):
                genome_data = self.get_bot_genome()
                genome_data = sorted(list(genome_data))
                genome_data = tuple(genome_data)
                if self.store_history:
                    if not (genome_data in list(self.__class__.scoreboard[self.population_id].keys())):
                        self.__class__.scoreboard[self.population_id].update({genome_data: self.score})
                    if not (genome_data in list(self.__class__.scoreboard[-1].keys())):
                        self.__class__.scoreboard[-1].update({genome_data: self.score})

                genome_data = {genome_data}
                history = history.union(genome_data)

                self.__class__.bots_history[self.population_id] = history
                global_history = Bot.get_bots_history(-1)
                global_history = global_history.union(genome_data)
                self.__class__.bots_history[-1] = global_history
            else:
                Bot.set_bots_flag_done(self.population_id, True)
            # print(self.__class__.bots_history[self.population_id])
        pass

    def get_bot_score(self) -> int or float:
        return self.score

    def get_bot_genome(self) -> list:
        """
        Get genome data from bot

        Returns
        -------
        genome_data: tuple      genome data. Collected name and value from each gene
        """
        genome_data = []
        for idx in range(len(self.bot_genome)):
            # noinspection PyUnresolvedReferences
            genome_data.append((self.bot_genome[idx].gene_name, self.bot_genome[idx].itself))
        # genome_data = sorted(genome_data)
        # genome_data = tuple(genome_data)
        return genome_data

    def get_bot_genes_data(self) -> object:
        """
        Get genome data from bot

        Returns
        -------
        genome_data: np.array      genome data. Collected values from each gene

        """
        genes_data = []
        for idx in range(len(self.bot_genome)):
            genes_data.append(self.bot_genome[idx].itself)
        return np.asarray(genes_data)

    def is_it_dupe_bot(self) -> bool:
        """
        Check is this bot in history?

        Returns
        -------
        result: bool    True if this bot have duped genome
        """
        result = False
        if self.share_history:
            genome_data = self.get_bot_genome()
            genome_data = sorted(list(genome_data))
            genome_data = tuple(genome_data)
            if genome_data in self.__class__.bots_history[-1]:
                result = True
                self.__class__.dupe_counter[-1] += 1
            if genome_data in self.__class__.bots_history[self.population_id]:
                result = True
                self.__class__.dupe_counter[self.population_id] += 1
        return result

    def re_generate_bot(self) -> None:
        """
        re-generate the genes value of ALL genes if not self.bot_freeze == True

        Returns
        -------
        None
        """
        done = True
        counter = 0
        self.add_bot_genome_to_history()
        if not self.bot_freeze:
            while done:
                counter += 1
                for idx in range(self.genes_qty):
                    # noinspection PyUnresolvedReferences
                    self.bot_genome[idx].re_generate_gene()
                if self.share_history:
                    done = self.is_it_dupe_bot()
                else:
                    done = False
        pass

    def mutate_bot(self) -> None:
        """
        Mutate (re_generate) bot genes with mutation_probability * self.genes_qty
        if not self.bot_freeze == True

        Returns
        -------
        None
        """
        done = True
        counter = 0
        self.add_bot_genome_to_history()
        if not self.bot_freeze:
            while done:
                counter += 1
                mutate_genes_num = int(round(self.bot_mutation_probability * self.genes_qty, 0))
                if mutate_genes_num == 0:
                    mutate_genes_num = 1
                indices = np.random.choice(self.genes_qty, mutate_genes_num, replace=False)
                for idx in indices:
                    self.bot_genome[idx].mutate_gene()
                if self.share_history:
                    done = self.is_it_dupe_bot()
                else:
                    done = False
        pass

    def mutate_genes(self, indices) -> None:
        """
        Parameters
        ----------
        indices: int    genes indexes to mutate
        """
        if not self.bot_freeze:
            for idx in indices:
                self.bot_genome[idx].mutate_gene()
        pass

    def set_genes_freeze(self, indices) -> None:
        """
        Parameters
        ----------
        indices: int    genes indexes to freeze
        """
        for idx in indices:
            self.bot_genome[idx].freeze_gene()
        pass

    def set_genes_unfreeze(self, indices) -> None:
        """
        Parameters
        ----------
        indices: int    genes indexes to unfreeze
        """
        for idx in indices:
            # noinspection PyUnresolvedReferences
            self.bot_genome[idx].unfreeze_gene()
        pass

    def freeze_bot(self) -> None:
        """ Freeze the bot from mutation and re-generation
        Returns
        -------
        None
        """
        self.bot_freeze = True
        pass

    def unfreeze_bot(self) -> None:
        """ unFreeze the bot for  mutation and re-generation
        Returns
        -------
        None
        """
        self.bot_freeze = False
        pass


class FillBag:
    """
    Class for genetic algorithm  - set the capacity and fill the FillBag with int or float items
    """

    def __init__(self, bag_id: int, capacity: int or float, name: str = None) -> None:
        """
        Args:
           bag_id (int):            id of bag
           capacity (int or float): bag capacity
           name (str):              name of FillBag (default: name = None)
        Returns:
            None
        """
        self.name = name
        self.bag_id: int = bag_id
        self.capacity: int or float = capacity
        self.volume: list = []
        self.score: float = np.NINF
        pass

    def append_to_bag(self, weight: int or float):
        """
        Append weight to the FillBag

        Args:
            weight (int or float):  weight for fill the Bag
        """
        self.volume.append(weight)
        # self.calc_bag_score()
        pass

    def calc_bag_score(self) -> None:
        """
        Get score of the bag
        """
        check_weights_in = abs(self.capacity - sum(self.volume))
        if round(check_weights_in, 0) != 0:
            self.score = -(check_weights_in ** 2)
        else:
            self.score = 0
        pass

    def reset_bag(self) -> None:
        """ Reset the FillBag score and volume

        Returns:
            None
        """
        self.volume = []
        self.score = np.NINF
        pass


class Shelf:
    """
    Class for working with FillBag class.
    Use it with FillBag class if you need to fill any qty of weighted items to containers with capacity/volume
    """

    def __init__(self,
                 bags_capacity: list,
                 to_bags_items: list,
                 shelf_id: int = 0,
                 ) -> None:
        """ Class initialization

        Args:
            bags_capacity (list):       bags capacity list
            to_bags_items (list):       list of items to fill in bag
            shelf_id (int):             id of shelf

        Returns:
            None
        """
        self.bags_capacity: list = bags_capacity
        self.to_bags_items: list = to_bags_items
        self.bags_qty: int = len(bags_capacity)
        self.shelf_id: int = shelf_id
        self.shelf_items: {int: FillBag} = {}
        self.score: int or float = np.NINF
        self.last_bot_genes: list = []
        self.best_bot = Bot
        self.last_bot_genome: list = []
        self._init_shelf_items()
        pass

    def _init_shelf_items(self):
        for bag_capacity, bag_id in zip(self.bags_capacity, range(self.bags_qty)):
            self.shelf_items.update({bag_id: FillBag(bag_id, bag_capacity)})
        pass

    def _calc_shelf_score(self) -> None:
        shelf_score = []
        for bag in self.shelf_items.values():
            bag.calc_bag_score()
            shelf_score.append(bag.score)
        self.score = sum(shelf_score)
        pass

    def fill_bags_on_shelf(self, genome: list) -> None:
        """
        Fill the bags with genes_data
        Args:
            genome (list):  list of items to fill it to bags
        """
        assert len(self.to_bags_items) == len(genome), "Q-ty of fill items and q-ty of genes must be equal"

        self.last_bot_genome = genome
        for one_gene_idx, (one_gene_name, one_gene_data) in enumerate(genome):
            self.shelf_items[one_gene_data].append_to_bag(self.to_bags_items[one_gene_idx])
        self._calc_shelf_score()
        pass

    def clean_shelf(self) -> None:
        """ Cleaning the shelf of bags. Reset all bags"""
        for bag in self.shelf_items.values():
            bag.reset_bag()
        pass

    def show_best_result(self, best_bot: Bot) -> None:
        """
        Show best bot result

        Args:
            best_bot (object): Bot class object
        Returns:
            None
        """
        print(f'Sum of all bags capacities: {sum(self.bags_capacity)}')
        print(f'Sum of all items for bags: {sum(self.to_bags_items)}')
        msg = str()
        for train_idx, train_cap in enumerate(self.bags_capacity):
            pool = []
            for gene_idx in range(len(self.to_bags_items)):
                train_num = best_bot.bot_genome[gene_idx].itself
                if train_idx == train_num:
                    pool.append(self.to_bags_items[gene_idx])
            msg = f'{msg}Bag:{train_idx + 1:02d} ({train_cap}-{pool}**2={-((train_cap - sum(pool)) ** 2)}  '
        print(f'{msg} Total: {best_bot.score}')
        self.best_bot = best_bot
        pass

    def get_best_bot(self):
        """
        Returns:
            best_bot (Bot):     return Bot class object
        """
        return self.best_bot


class TimeBag(FillBag):
    """ Class for genetic algorithm, fill the TimeBag with time range and "weight" for this time range """

    def __init__(self,
                 bag_id: int,
                 bag_name: str,
                 worktime: (str, str),
                 capacity: int or float,
                 one_unit: int = 3600,
                 time_format: str = "%Y-%m-%dT%H:%M:%S%z"
                 ) -> None:
        """ Class initialization
        Args:
            bag_id (int):               id of bag
            bag_name (str):             name of the bag
            worktime (str, str):        (start_time, stop_time) start and stop time
                                        as str ("2015-03-22T10:00:00+0900" or "18:00") do not mix format
            capacity (int or float):    bag capacity
            one_unit (int):             one_unit in seconds. by default one_unit = 3600sec (1h)
                                        one_unit using for calculating score

        Returns:
            None
        """
        super().__init__(bag_id, capacity)
        self.bag_name = bag_name
        self.time_format = time_format
        if self.time_format == "%H:%M":
            self.worktime = (datetime.datetime.strptime(worktime[0], self.time_format).time(),
                             datetime.datetime.strptime(worktime[1], self.time_format).time())
            self.worktime = (datetime.datetime.combine(datetime.date.today(), self.worktime[0]),
                             datetime.datetime.combine(datetime.date.today(), self.worktime[1]))
        else:
            self.worktime = (datetime.datetime.strptime(worktime[0], self.time_format),
                             datetime.datetime.strptime(worktime[1], self.time_format))

        self.worktime_range = DateTimeRange(worktime[0], worktime[1])
        self.worktime_range.start_time_format = self.time_format
        self.worktime_range.end_time_format = self.time_format
        self.one_unit: int = one_unit
        self.work_duration = self.worktime_range.get_timedelta_second() / self.one_unit
        self.time_ranges: list = []
        self.max_weights: list = []
        self.time_start_time_end: list = []
        pass

    def append_to_bag(self, range_and_weight: tuple) -> None:
        """Append range of time and "weight" to the TimeBag

        Args:
            range_and_weight ((str, str, int or float)):   range, weight and max_weight to append to TimeBag
        Returns:
            None
        """
        self.time_start_time_end.append((range_and_weight[0], range_and_weight[1]))
        if not (range_and_weight[0] is None) and not (range_and_weight[1] is None):
            time_range = DateTimeRange(range_and_weight[0], range_and_weight[1])
            time_range.start_time_format = self.time_format
            time_range.end_time_format = self.time_format
        else:
            time_range = None
        self.volume.append(range_and_weight[2])
        self.time_ranges.append(time_range)
        self.max_weights.append(range_and_weight[3])
        # self._calc_bag_score()
        pass

    def _calc_worktime_and_time_periods_score(self) -> int:
        time_ranges_score_lst = [0 for _ in range(len(self.time_ranges))]
        correct_time_ranges_indices = []
        # ranges_duration = 0
        for idx, time_range in enumerate(self.time_ranges):
            if time_range is not None:
                if not time_range.is_valid_timerange():
                    time_ranges_score_lst[idx] += -48
                else:
                    if self.time_start_time_end[idx][0] == self.time_start_time_end[idx][1]:
                        time_ranges_score_lst[idx] += -12
                    if not (self.worktime[0] <= self.time_start_time_end[idx][0] <= self.worktime[1]) or \
                            not (self.worktime[0] < self.time_start_time_end[idx][1] <= self.worktime[1]):
                        time_ranges_score_lst[idx] += -48
                    else:
                        correct_time_ranges_indices.append(idx)
                    if self.volume[idx] > self.max_weights[idx]:
                        penalty = self.max_weights[idx] - self.volume[idx]
                        time_ranges_score_lst[idx] += -12 * abs(penalty)
            else:
                # time_range is not valid cos of None
                time_ranges_score_lst[idx] += -14

        time_pairs_combs_indices = combinations(correct_time_ranges_indices, 2)
        for idx_pair in time_pairs_combs_indices:
            # checking if ranges have same time range
            if self.time_ranges[idx_pair[0]] == self.time_ranges[idx_pair[1]]:
                time_ranges_score_lst[idx_pair[0]] += self.volume[idx_pair[0]] * -24
                time_ranges_score_lst[idx_pair[1]] += self.volume[idx_pair[1]] * -24

            # checking if ranges have intersection in time range
            time_period_truncated = copy.copy(self.time_ranges[idx_pair[1]])
            time_period_truncated.truncate(2)
            if self.time_ranges[idx_pair[0]].is_intersection(time_period_truncated):
                time_intersect = self.time_ranges[idx_pair[0]].intersection(self.time_ranges[idx_pair[1]])
                overlap_time = time_intersect.timedelta.total_seconds() // self.one_unit
                time_ranges_score_lst[idx_pair[0]] += overlap_time * -12
                time_ranges_score_lst[idx_pair[1]] += overlap_time * -12
            pass

        # check if None exist but we need a period
        len_time_ranges_score_lst = len(time_ranges_score_lst)
        sum_time_ranges_score_lst = sum(time_ranges_score_lst)
        for idx, time_range_score in enumerate(time_ranges_score_lst):
            if time_range_score == -14 and len_time_ranges_score_lst > 1 and \
                    sum_time_ranges_score_lst == -14 and self.score == 0:
                time_ranges_score_lst[idx] = 0
                sum_time_ranges_score_lst = sum(time_ranges_score_lst)
            elif time_range_score == -14 and len_time_ranges_score_lst == 1:
                time_ranges_score_lst[idx] += 48

        score = sum_time_ranges_score_lst
        return score

    def calc_bag_score(self) -> None:
        """ Calculate the score of the bag

        Returns:
            None
        """
        # check capacity and set score for capacity
        check_weights_in = self.capacity - sum(self.volume)
        if check_weights_in > 0:
            self.score = -(abs(check_weights_in * 9))
        elif check_weights_in < 0:
            self.score = -(abs(check_weights_in * 6))
        else:
            self.score = 0

        # check worktime and periods
        self.score += self._calc_worktime_and_time_periods_score()
        pass

    def reset_bag(self) -> None:
        """
        Reset the TimeBag, score, volume and time_ranges

        Returns:
            None
        """
        self.time_ranges = []
        self.volume = []
        self.max_weights = []
        self.time_start_time_end = []
        self.score = np.NINF
        pass


class TimePlan:
    """
    Class for working with TimeBag objects.
    Use it with TimeBag class if u need to calculate resource planning for a period
    """

    def __init__(self,
                 time_bags_configs: dict,
                 to_bags_items: dict,
                 unit: int = 3600,
                 time_plan_id: int = 0,
                 time_format: str = "%H:%M"
                 ) -> None:

        """
        Initialization of TimePlan class

        Args:
            time_bags_configs (dict): '2015-03-22T10:00:00+0900' or '18:00' do not mix format capacity -
                                        int or float volume of the time_bag
            to_bags_items (dict):       items for bags. Check the examples above
            unit (int):                 base unit for calculate score in seconds (default - 3600 seconds (1h))
            time_plan_id (int):         TimePlan id

        Example:
            time_bags_capacity = {"Monday": ("09:00", "18:00", 9), "Tuesday": ("2021-02-16T09:00:00+0300", "2021-02-16T21:00:00+0300", 12)}
            to_bags_items = {"Person_id": {"Monday": ("09:00", "15:00", 6), "Wednesday": (start_time, stop_time, weight)}}
        """

        self.time_bags_configs: dict = time_bags_configs
        self.to_bags_items: dict = to_bags_items
        self.unit: int = unit
        self.time_plan_id: int = time_plan_id
        self.time_format = time_format
        self.bags_qty: int = len(self.time_bags_configs.keys())
        self.time_plan_items: {str: TimeBag} = {}
        self.score: int or float = np.NINF
        self.genes: dict = {}
        self.genes_time_iterations: dict = {}
        self.genes_max_weight: dict = {}
        self.bags_names: list = []
        self.output_col_names: list = []
        self.best_bot = object
        self._init_time_plan_items()
        pass

    @staticmethod
    def get_time_iterations(time_period: tuple, seconds_unit=3600) -> list:
        """
        Get time iterations from period using self.unit as step

        Args:
            time_period (tuple):    (str, str) time in string
            seconds_unit (int):     time unit in seconds

        Returns:
            time_iterations (list): iterations in period
        """
        time_range = DateTimeRange(time_period[0], time_period[1])
        if not time_range.is_valid_timerange():
            msg = f'Error: Time period {time_period} is not valid'
            sys.exit(msg)
        time_iterations: list = [None, ]
        for time_slice in time_range.range(datetime.timedelta(seconds=seconds_unit)):
            time_iterations.append(time_slice)
        return time_iterations

    def _init_time_plan_items(self):
        for bag_id, (bag_name, bag_config) in enumerate(self.time_bags_configs.items()):
            self.time_plan_items.update({bag_name: TimeBag(bag_id,
                                                           bag_name,
                                                           (bag_config[0], bag_config[1]),
                                                           bag_config[2],
                                                           time_format=self.time_format
                                                           )
                                         })
            self.output_col_names.append(f'{bag_name} {bag_config[0]}-{bag_config[1]} {bag_config[2]}')
            self.bags_names.append(bag_name)

        for person_name, tag_data in self.to_bags_items.items():
            for tag_name, tag_tuple in tag_data.items():
                if tag_name in self.bags_names:
                    gene_base_name = f'{person_name}_{tag_name}'
                    gene_name_start_time = f'{gene_base_name}_start_time'
                    gene_name_stop_time = f'{gene_base_name}_stop_time'

                    time_iterations = self.get_time_iterations((tag_tuple[0], tag_tuple[1]))
                    len_time_iterations = len(time_iterations)
                    # 0(zero) idx in time iterations - None. start and stop idxs starting from 1
                    self.genes.update({gene_name_start_time: [ix for ix in range(len_time_iterations)]})
                    self.genes.update({gene_name_stop_time: [ix for ix in range(len_time_iterations)]})

                    self.genes_time_iterations.update({gene_base_name: time_iterations})
                    self.genes_max_weight.update({gene_base_name: [tag_tuple[2]]})
            pass
        pass

    def fill_bags_on_shelf(self, genome: list) -> None:
        """
        Fill the bags with genes_data
        Args:
            genome (list):  list of items to fill it to bags
        """
        tuples_for_bags = [genome[ix:ix + 2] for ix in range(0, len(genome), 2)]
        for bag_tuple in tuples_for_bags:
            tag_name = bag_tuple[0][0]
            tag_name = tag_name.split('_')[-3]
            gene_base_name = bag_tuple[0][0]
            gene_base_name = gene_base_name.split(tag_name)[0]
            gene_base_name = f'{gene_base_name}{tag_name}'
            time_iterations = self.genes_time_iterations.get(gene_base_name)
            max_item_weight = self.genes_max_weight.get(gene_base_name)[0]
            range_start_idx = bag_tuple[0][1]
            range_end_idx = bag_tuple[1][1]
            item_weight = 0
            if range_start_idx != 0 and range_end_idx != 0:
                item_weight = (range_end_idx - range_start_idx) * 1
            bag_item = (time_iterations[range_start_idx], time_iterations[range_end_idx], item_weight, max_item_weight)
            if tag_name in self.bags_names:
                self.time_plan_items[tag_name].append_to_bag(bag_item)
        self._calc_shelf_score()
        pass

    def _calc_shelf_score(self) -> None:
        shelf_score = []
        for bag in self.time_plan_items.values():
            bag.calc_bag_score()
            if bag.score != np.NINF:
                shelf_score.append(bag.score)
        self.score = sum(shelf_score)
        pass

    def clean_shelf(self) -> None:
        """ Cleaning the shelf of bags. Reset all bags"""
        for bag in self.time_plan_items.values():
            bag.reset_bag()
        pass

    def show_best_result(self, best_bot: Bot) -> None:
        """
        Show best bot result

        Args:
            best_bot (object): Bot class object
        Returns:
            None
        """
        # print(f'Sum of all bags capacities: {self.time_bags_configs}')
        # print(f'Sum of all items for bags: {self.to_bags_items}')
        msg = str()
        genome = best_bot.get_bot_genome()
        tuples_for_bags = [genome[ix:ix + 2] for ix in range(0, len(genome), 2)]
        df_columns = list(self.time_bags_configs.keys())
        df_columns.insert(0, 'Names')
        persons_names = list(self.to_bags_items.keys())
        df = pd.DataFrame(index=pd.RangeIndex(0, len(persons_names)), columns=df_columns)
        df.loc[pd.RangeIndex(0, len(persons_names)), 'Names'] = persons_names

        bag_tuple: list
        for bag_tuple in tuples_for_bags:
            tag_name = bag_tuple[0][0]
            tag_name = tag_name.split('_')[-3]
            person_name = bag_tuple[0][0]
            person_name = person_name.split(tag_name)[0][:-1]
            gene_base_name = f'{person_name}_{tag_name}'

            time_iterations = self.genes_time_iterations.get(gene_base_name)
            bag_weight = (bag_tuple[1][1] - bag_tuple[0][1]) * 1
            time_start = time_iterations[bag_tuple[0][1]]
            time_end = time_iterations[bag_tuple[1][1]]

            time_range = DateTimeRange(time_start, time_end)
            time_range.start_time_format = self.time_format
            time_range.end_time_format = self.time_format

            time_period = f'{time_range}, {bag_weight:.1f}'
            if (time_start is not None) and (time_end is not None):
                df.loc[persons_names.index(person_name), tag_name] = time_period
            else:
                df.loc[persons_names.index(person_name), tag_name] = f'None'

        for (old_name, new_name) in zip(self.bags_names, self.output_col_names):
            df = df.rename(columns={old_name: new_name}, errors="raise")
        print(df.to_string())
        print(f'{msg} Total: {best_bot.score}')
        self.best_bot = best_bot
        pass

    def get_best_bot(self):
        """
        Returns:
            best_bot (Bot):     return Bot class object
        """
        return self.best_bot


class Couch:
    """ Couch class for train genetics algorithms """

    def __init__(self,
                 bags_configs: (dict or list),
                 to_bags_items: (dict or list),
                 score_condition: (int or float) = 0,
                 epoch_condition: int = 1000,
                 steps_to_evaluate: int = 40,
                 root_gene: str = "NORMAL",
                 typeof: str = 'fill_bag',
                 verbose: int = 0) -> None:
        """
        Class initializations

        Args:
            bags_configs (dict or list):    bags configs
            to_bags_items (dict or list):   items to fill in bags
            score_condition (int or float): score_condition (default: score_condition = 0,
            epoch_condition (int):          epoch_condition (default: epoch_condition = 1000)
            steps_to_evaluate (int):        steps_to_evaluate for remove population with lowscore
                                            (default: steps_to_evaluate = 40)
            root_gene (str or dict):        root_gene initialization ("NORMAL", LONGEST or root_dict)
                                            for dividing populations
            typeof (str):                   type of genetic build, valid options: "fill_bag", time_bag", "simple_genes"
            verbose (int):                  details printed during calculations

        Returns:
            None
        """
        self.typeof = typeof
        self.bags_configs = bags_configs
        self.to_bags_items = to_bags_items
        self.score_condition: (int or float) = score_condition
        self.epoch_condition: int = epoch_condition
        self.steps_to_evaluate: int = steps_to_evaluate
        self.bags_qty = len(self.bags_configs)
        self.genes: dict = {}
        self.root_genes = root_gene
        self.scoring: object = object
        if self.typeof == "fill_bag":
            self._init_fill_bags()
        elif self.typeof == "time_bag":
            self._init_time_bags()
        self.mega_population = MegaPopulation(populations_qty=self.bags_qty,
                                              population_size=500,
                                              genes=self.genes,
                                              root_genes=self.root_genes,
                                              scoring=self.scoring,
                                              score_condition=self.score_condition,
                                              epoch_condition=self.epoch_condition,
                                              steps_to_evaluate=self.steps_to_evaluate,
                                              share_history=False,
                                              store_history=False,
                                              sorting_reverse=False,
                                              parents_qty=2,
                                              survival_probability=0.2,
                                              population_mutation_probability=0.30,
                                              bot_mutation_probability=0.15,
                                              gene_mutation_probability=0.3,
                                              verbose=verbose)

        pass

    def _init_fill_bags(self) -> None:
        assert isinstance(self.bags_configs, list), f"Error: with typeof {self.typeof} bags_configs MUST be a list " \
                                                    f"with capacity of each bag "
        assert isinstance(self.to_bags_items, list), f"Error: with typeof {self.typeof} to_bags_items MUST be a list " \
                                                     f"with weights items "
        for ix in range(len(self.to_bags_items)):
            bags_nums = [ix for ix in range(self.bags_qty)]
            self.genes.update({f'{ix + 1:002d}': bags_nums})
        self.scoring = Shelf(self.bags_configs, self.to_bags_items)
        self.possible_iterations, _ = get_iter_count(self.genes, {})
        msg = f'Starting GA for searching of the best fill of capacity:\nCapacity: {self.bags_configs}\n' \
              f'Items weight: {self.to_bags_items}\n'
        msg += f'Total possible iterations: {self.possible_iterations:,}\n'
        msg += f'Total genes qty: {len(self.genes)}'
        print(msg)
        pass

    def _init_time_bags(self) -> None:
        assert isinstance(self.bags_configs, dict), f"Error: with typeof {self.typeof} bags_configs MUST be a dict " \
                                                    f"with config of each bag "
        assert isinstance(self.to_bags_items, dict), f"Error: with typeof {self.typeof} to_bags_items MUST be a dict " \
                                                     f"with time items "

        self.scoring = TimePlan(self.bags_configs, self.to_bags_items)
        self.genes = self.scoring.genes
        self.possible_iterations, self.iterations_items = get_iter_count(self.genes, {})
        all_bags_iter = []
        for idx in range(0, len(self.iterations_items), 2):
            one_bag_iter = self.iterations_items[idx] ** 2
            all_bags_iter.append(one_bag_iter)
        self.possible_iterations = reduce(operator.mul, all_bags_iter, 1)
        msg = f'Starting GA for searching of the best TimePlan.\n'
        msg += f'Total possible iterations: {self.possible_iterations:,}\n'
        msg += f'Total genes qty: {len(self.genes)}'
        print(msg)
        pass

    def train(self) -> None:
        """ Train megapopulations """
        self.mega_population.train_populations()
        pass


# TODO save and load genetics params and history

class Population:
    """
    Class for creating a population of bots,
    with methods to train, breed, mutate, show and check rules population of bots
    """

    def __init__(self,
                 size: int,
                 genes: dict,
                 population_id: int,
                 root_gene: dict,
                 scoring: object,
                 share_history: bool = False,
                 store_history: bool = False,
                 sorting_reverse: bool = False,
                 parents_qty: int = 2,
                 survival_probability: float = 0.2,
                 population_mutation_probability: float = 0.2,
                 bot_mutation_probability: float = 0.2,
                 gene_mutation_probability: float = 0.10,
                 verbose: int = 5):

        """
        Args:
            size (int):
            genes (dict):
            population_id: int
            root_gene: dict
            share_history: bool
            store_history: bool
            sorting_reverse: bool
            parents_qty: int
            survival_probability: float
            population_mutation_probability: float
            bot_mutation_probability: float
            gene_mutation_probability: float
        """
        self.size: int = size
        self.genes: dict = genes
        self.population_id: int = population_id
        self.scoring: Shelf = scoring
        self.root_gene: dict = root_gene
        self.share_history: bool = share_history
        self.store_history: bool = store_history
        self.sorting_reverse: bool = sorting_reverse
        self.parents_qty: int = parents_qty
        self.survival_probability: float = survival_probability
        self.survival_qty = int(self.survival_probability * self.size)
        if self.survival_qty < 2:
            self.survival_qty = 2
        self.population_mutation_probability: float = population_mutation_probability
        self.bot_mutation_probability: float = bot_mutation_probability
        self.gene_mutation_probability: float = gene_mutation_probability
        self.verbose: int = verbose
        self.bots_pool: {int: Bot} = {}
        self.population_epoch: int = 0
        self.best_bots_indices: list = []
        self.highscore_score: float = np.NINF
        self.highscore_idx = None
        self.best_bot = Bot
        self.start_training_time = time.time()
        if self.sorting_reverse:
            self.highscore_score = np.PINF
        for idx in range(self.size):
            self.bots_pool.update({idx: Bot(self.population_id,
                                            self.genes,
                                            self.root_gene,
                                            self.share_history,
                                            self.store_history,
                                            self.bot_mutation_probability)
                                   })
        # setting sharing history bots index
        if self.share_history:
            Bot.set_bots_history(set(), -1)
            Bot.dupe_counter[-1] = 0
            Bot.scoreboard[-1] = {}

        if self.verbose == 0:
            self.show_epoch = 100
            self.show_best_bots = 5
        elif self.verbose == 1:
            self.show_epoch = 50
            self.show_best_bots = 5
        elif self.verbose == 2:
            self.show_epoch = 10
            self.show_best_bots = 5
        elif self.verbose == 3:
            self.show_epoch = 5
            self.show_best_bots = 5
        elif self.verbose == 4:
            self.show_epoch = 1
            self.show_best_bots = 5
        elif self.verbose == 5:
            self.show_epoch = 1
            self.show_best_bots = self.survival_qty
        self.solved = False
        pass

    def set_bot_score(self, idx, score: float):
        self.bots_pool[idx].score = score
        pass

    def get_best_bot_score_indices(self, reverse, best_qty):
        """
        Get the list of indexes for best bots

        Args:
            reverse (bool): sorting reverse True or False
            best_qty (int): best bots qty

        Returns:
            indices (list): list of bot indexes
        """
        score_list = [self.bots_pool[idx].score for idx in range(self.size)]
        score_list = np.asarray(score_list)
        indices = [x[0] for x in
                   sorted(enumerate(score_list), key=lambda x: x[1], reverse=reverse)[-best_qty:]]
        indices.reverse()
        return indices

    def re_generate(self, indices: list) -> None:
        """ re_generate bots

        Args:
            indices (list): bot indexes for mutation
        Returns:
              None
        """
        for idx in indices:
            self.bots_pool[idx].re_generate_bot()
        pass

    def mutate(self, indices: list) -> None:
        """ Mutate bots

        Args:
            indices (list): bot indexes for mutation
        Returns:
              None
        """
        for idx in indices:
            self.bots_pool[idx].mutate_bot()
        pass

    def generate_new_population(self):
        """ Generate new bots_pool from best bots (parents)
        but keep the all best bots in
        ~ breed new bots_pool (based on (1-self.population_mutation_probability))
        ~ mutate bots_pool (based on self.population_mutation_probability)
        Returns
        -------
        None
        """
        # best_bots_indices = self.get_best_bot_score_indices(reverse=self.sorting_reverse, best_qty=self.survival_qty)
        bots_to_change = [idx for idx in range(self.size) if idx not in self.best_bots_indices]

        changing_population_size = int(len(bots_to_change) * self.population_mutation_probability)
        mutation_population_size = changing_population_size // 2
        if mutation_population_size < 1:
            mutation_population_size = 1
        mutation_population_indices = random.sample(bots_to_change, mutation_population_size)
        self._mutate_best_bots_copy(self.best_bots_indices, mutation_population_indices)

        re_generation_population_size = changing_population_size - mutation_population_size
        if re_generation_population_size < 1:
            re_generation_population_size = 1
        re_generation_population_indices = random.sample(bots_to_change, re_generation_population_size)
        self.re_generate(re_generation_population_indices)

        bots_to_remove_indices = [idx for idx in bots_to_change if not (idx in mutation_population_indices) and not (
                idx in re_generation_population_indices)]
        self.breed_bots(self.best_bots_indices, bots_to_remove_indices)
        pass

    def _divide_genes(self, parents_genes_qty: dict):
        """ Dividing pool of genes for each parent with random indices

        Args:
            parents_genes_qty (dict):   key is idx number of parent, value - genes_qty

        Returns:
            parents_genes_selected (dict):  key is idx number of parent, value - gene_idx indices (list)
        """
        parents_genes_selected = dict()
        genes_set = set(range(len(self.genes)))
        for parent_num, parent_genes_qty in parents_genes_qty.items():
            parents_genes_selected[parent_num] = random.sample(genes_set, parent_genes_qty)
            genes_set = genes_set.difference(set(parents_genes_selected[parent_num]))
        return parents_genes_selected

    def _copy_bot_gene(self, source_bot_index: int, target_bot_index: int, gene_idx: int) -> None:
        """
        Copy a gene of source bot to target bot

        Args:
            source_bot_index : int      index of source bot
            target_bot_index : int      index of target bot
            gene_idx : int              gene index

        Returns:
            None
        """
        # adding target bot to history
        self.bots_pool[target_bot_index].bot_genome[gene_idx].gene_name = \
            self.bots_pool[source_bot_index].bot_genome[gene_idx].gene_name
        self.bots_pool[target_bot_index].bot_genome[gene_idx].itself = \
            self.bots_pool[source_bot_index].bot_genome[gene_idx].itself
        pass

    def _mutate_best_bots_copy(self, best_bot_indices: list, target_bot_indices: list) -> None:
        """
        Copy source bot to target bot

        Args:
            best_bot_indices (list):      indices of source bots
            target_bot_indices (list):    indices of target bots

        Returns:
            None
        """
        for bots_target in target_bot_indices:
            done = False
            while done:
                parents_indices = random.sample(best_bot_indices, 1)
                for gene_idx in range(len(self.genes)):
                    self._copy_bot_gene(parents_indices[0], bots_target, gene_idx)
                if self.share_history:
                    done = not (self.bots_pool[bots_target].is_it_dupe_bot())
                    if done:
                        self.bots_pool[bots_target].add_bot_genome_to_history()
                else:
                    done = False
            if random.random() < self.bot_mutation_probability:
                self.bots_pool[bots_target].mutate_bot()
        pass

    def breed_bots(self, best_bots_indices: list, bots_to_remove_indices: list) -> None:
        """
        Breed n-bots to one (n - self.parents_qty)

        Args:
            best_bots_indices (list):           list of parents indexes
            bots_to_remove_indices (list):      list of bots indexes to remove
        Returns:
            None
        """
        parent_genes_qty = dict()
        if len(self.genes) < self.parents_qty:
            msg = "Error: genes qty less than parents qty for breeding"
            sys.exit(msg)
        elif self.parents_qty < 2:
            msg = "Error: parents qty for breeding can't be less than 2"
            sys.exit(msg)
        else:
            for parent_num in range(self.parents_qty - 1):
                parent_genes_qty[parent_num] = int(len(self.genes) // self.parents_qty)
            parent_genes_qty[self.parents_qty - 1] = len(self.genes) - (len(self.genes) // self.parents_qty) * (
                    self.parents_qty - 1)
        for bots_to_change_idx in bots_to_remove_indices:
            done = True
            while done:
                parents_genes_selected = self._divide_genes(parent_genes_qty)
                parents_indices = random.sample(best_bots_indices, self.parents_qty)
                for parent_num, parent_idx in enumerate(parents_indices):
                    for gene_selected in parents_genes_selected[parent_num]:
                        self._copy_bot_gene(parent_idx, bots_to_change_idx, gene_selected)
                        if random.random() < self.bot_mutation_probability:
                            # noinspection PyUnresolvedReferences
                            self.bots_pool[bots_to_change_idx].bot_genome[gene_selected].re_generate_gene()
                if self.share_history:
                    # noinspection PyUnresolvedReferences
                    done = self.bots_pool[bots_to_change_idx].is_it_dupe_bot()
                    if not done:
                        # noinspection PyUnresolvedReferences
                        self.bots_pool[bots_to_change_idx].add_bot_genome_to_history()
                else:
                    done = False
        pass

    def train_population(self):
        """ Train population, calculate highscore and store highscore and index of highscore bot """
        for bot_id in range(self.size):
            # noinspection PyUnresolvedReferences
            self.scoring.fill_bags_on_shelf(self.bots_pool[bot_id].get_bot_genome())
            score = self.scoring.score
            if self.sorting_reverse:
                if self.highscore_score > score:
                    self.highscore_score = score
                    self.highscore_idx = bot_id
            else:
                if self.highscore_score < score:
                    self.highscore_score = score
                    self.highscore_idx = bot_id
            self.set_bot_score(bot_id, score)
            self.scoring.clean_shelf()
        pass

    @staticmethod
    def show_history():
        """ Show history of all populations """
        for params, score in Bot.scoreboard[-1].items():
            print(f'{score:3d}:', *params)
        pass

    def check_rules(self, score_condition) -> None:
        """ Checking the rules """
        self.best_bots_indices = self.get_best_bot_score_indices(reverse=self.sorting_reverse,
                                                                 best_qty=self.survival_qty)
        short_best_bots_indices = self.best_bots_indices[:self.show_best_bots]
        if self.population_epoch % self.show_epoch == 0:
            msg = f'id: {self.population_id:03d} Epoch: {self.population_epoch:06d}, '
            for best_bot_idx in short_best_bots_indices:
                msg = f'{msg} {self.bots_pool[best_bot_idx].score}'
            print(msg)
        self.best_bot = self.bots_pool[short_best_bots_indices[0]]

        if (self.highscore_score >= score_condition) or Bot.get_bots_flag_done(self.population_id):
            msg = f'>>> Training ends: {datetime_now()}\n'
            msg += f'Total training time: {(time.time() - self.start_training_time) / 60:.1f} min\n'
            msg += f'Solved in {self.population_epoch} epochs,\nbot with highest score & params:\n'
            msg += f'id: {self.population_id:03d} Epoch: {self.population_epoch:06d}, {self.highscore_score} ' \
                   f'{self.bots_pool[self.highscore_idx].get_bot_genome()}'
            print(msg)
            self.scoring.show_best_result(self.best_bot)
            self.solved = True
        pass

    def population_step(self, score_condition) -> None:
        """
        Population one step
        """
        self.train_population()
        self.check_rules(score_condition)
        self.generate_new_population()
        self.population_epoch += 1
        pass


class MegaPopulation:
    def __init__(self,
                 populations_qty: int,
                 population_size: int,
                 genes: dict,
                 root_genes: dict or str or tuple,
                 scoring: object = object,
                 score_condition: (int or float) = 0,
                 epoch_condition: int = 1000,
                 steps_to_evaluate: int = 40,
                 share_history: bool = False,
                 store_history: bool = False,
                 sorting_reverse: bool = False,
                 parents_qty: int = 3,
                 survival_probability: float = 0.2,
                 population_mutation_probability: float = 0.3,
                 bot_mutation_probability: float = 0.2,
                 gene_mutation_probability: float = 0.25,
                 verbose: int = 1):

        self.populations_qty: int = populations_qty
        self.population_size: int = population_size
        self.genes: dict = genes
        self.genes_qty = len(self.genes)
        self.root_genes = root_genes
        self.scoring = scoring
        self.score_condition: (int or float) = score_condition
        self.epoch_condition: int = epoch_condition
        self.steps_to_evaluate: int = steps_to_evaluate
        self.share_history: bool = share_history
        self.store_history: bool = store_history
        # self.population_id: int = population_id
        self.sorting_reverse: bool = sorting_reverse
        self.parents_qty: int = parents_qty
        self.survival_probability: float = survival_probability
        self.population_mutation_probability: float = population_mutation_probability
        self.bot_mutation_probability: float = bot_mutation_probability
        self.gene_mutation_probability: float = gene_mutation_probability
        self.verbose = verbose
        self.populations = dict()
        self.population_epoch: int = 0
        self.possible_iterations, self.genes_iter_count = get_iter_count(self.genes, {})
        if isinstance(self.root_genes, str):
            if self.root_genes == 'NORMAL':
                self._init_populations(init='NORMAL')
            elif self.root_genes.upper() == 'LONGEST':
                self._init_populations(init='LONGEST')
            else:
                msg = f'Warning:  unknown type of root genes {self.root_genes}. Using LONGEST'
                print(msg)
                self._init_populations(init='LONGEST')
        elif isinstance(self.root_genes, dict):
            self._init_populations(init='ROOT')
        else:
            self._init_populations(init='ROOT')
        pass

    def _init_populations(self, init='LONGEST'):
        """
        Initialization of populations depends on options
        """
        if init == 'LONGEST':
            self._choose_root_genes()
            self._root_genes_init()
        elif init == 'NORMAL':
            self._normal_init()
        elif init == 'ROOT':
            self.root_genes_qty = len(self.root_genes)
            self._root_genes_init()
        else:
            msg = f"Warning: unknown init type {init}"
            sys.exit(msg)
        pass

    def _choose_root_genes(self):
        self.root_genes = dict()
        root_genes_qty = int(len(self.genes) // 10)
        indices = [x[0] for x in sorted(enumerate(self.genes_iter_count), key=lambda x: x[1])]
        indices.reverse()
        for idx, (gene_name, gene_items) in enumerate(self.genes.items()):
            if idx in indices[:root_genes_qty]:
                self.root_genes.update({gene_name: gene_items})
        pass

    def _create_population(self, idx, root_gene):
        self.populations[idx] = Population(self.population_size,
                                           self.genes,
                                           population_id=idx,
                                           root_gene=root_gene,
                                           scoring=self.scoring,
                                           share_history=self.share_history,
                                           store_history=self.store_history,
                                           sorting_reverse=self.sorting_reverse,
                                           parents_qty=self.parents_qty,
                                           survival_probability=self.survival_probability,
                                           population_mutation_probability=self.population_mutation_probability,
                                           bot_mutation_probability=self.bot_mutation_probability,
                                           gene_mutation_probability=self.gene_mutation_probability,
                                           verbose=self.verbose)
        pass

    def _normal_init(self):
        root_gene = {}
        for idx in range(self.populations_qty):
            self._create_population(idx, root_gene)
        pass

    def _root_genes_init(self):
        self.populations_qty = 0
        idx = 0
        for gene_name in self.root_genes.keys():
            gene_iter, gene_items = count_params(self.root_genes.get(gene_name))
            self.populations_qty += gene_iter
            for gene_item in gene_items:
                self._create_population(idx, {gene_name: gene_item})
                idx += 1
        self.populations_qty = idx
        pass

    def show_best_bot(self):
        pass

    def train_populations(self):
        done = False
        idx_wheel = list(range(self.populations_qty))
        idx_wheel_len = self.populations_qty
        populations_score = dict()
        idx_cycler = cycle(idx_wheel)
        pop_count = 0
        lowscore = np.PINF
        lowscore_idx = None

        print(">>> Training starts at ", datetime_now())
        start_training_time = time.time()
        for idx in idx_wheel:
            self.populations[idx].start_training_time = start_training_time
        while not done:
            if len(idx_wheel) == idx_wheel_len:
                idx = next(idx_cycler)
                self.populations[idx].population_step(self.score_condition)
                if self.populations[idx].population_epoch % self.steps_to_evaluate == 0:
                    score_lst = []
                    for ix, best_bot_idx in enumerate(self.populations[idx].best_bots_indices):
                        score = self.populations[idx].bots_pool[best_bot_idx].score
                        score_lst.append(score)
                        # Calculate score on 5 best population bots
                        if ix >= 5:
                            break
                    populations_score[idx] = sum(score_lst)
                    if lowscore > populations_score[idx]:
                        lowscore = populations_score[idx]
                    pop_count += 1
                    if (pop_count == idx_wheel_len) and (idx_wheel_len != 1):
                        lowscore = np.PINF
                        for key_idx, pop_score in populations_score.items():
                            if lowscore > pop_score:
                                lowscore = pop_score
                                lowscore_idx = key_idx
                        idx_wheel.remove(lowscore_idx)
                        msg = f"Removed population {lowscore_idx} with lowscore = {lowscore}. " \
                              f"Best bot score {self.populations[lowscore_idx].best_bot.score:.0f}"
                        print(msg)
                # epoch += 1
                done = self.populations[idx].solved
                if (self.epoch_condition <= self.populations[idx].population_epoch) and \
                        (idx == idx_wheel[-1]):
                    best_score = np.NINF
                    best_ix: int = 0
                    for ix in idx_wheel:
                        if self.populations[ix].best_bot.score > best_score:
                            best_score = self.populations[ix].best_bot.score
                            best_ix = ix
                    msg = f'>>> Training ends: {datetime_now}\n'
                    msg += f'Total training time: {(time.time() - start_training_time) / 60:.1f} min\n'
                    msg += f'Stopped at {self.populations[best_ix].population_epoch - 1} ' \
                           f'epochs,\nbot with highest score & params:\n'
                    msg += f'id: {best_ix:03d} Epoch: {self.populations[best_ix].population_epoch - 1:05d}, ' \
                           f'{self.populations[best_ix].highscore_score} ' \
                           f'{self.populations[best_ix].best_bot.get_bot_genome()}\n'
                    print(msg)
                    self.populations[best_ix].scoring.show_best_result(self.populations[best_ix].best_bot)
                    done = True
            else:
                print('Reset lowscore')
                pop_count = 0
                lowscore = np.PINF
                lowscore_idx = None
                populations_score = dict()
                idx_wheel_len = len(idx_wheel)
                idx_cycler = cycle(idx_wheel)
        pass


if __name__ == '__main__':
    timezone = pytz.timezone("Europe/Moscow")

    def shelf_test():
        """
        Testing function for FillBag, Shelf and Couch classes

        Returns:
            None
        """
        trains_capacity = [1, 12, 72, 4, 55, 1, 11,
                           15, ]
        # 30,
        # 40]
        loads_for_trains = [1, 1, 1, 1, 4, 3, 3, 3, 1, 27,
                            31, 7, 7, 11, 22, 7, 7, 8, 4, 7,
                            5, 5, 5, ]
        # 10, 10, 10,
        # 8, 8, 8, 8, 8]

        couching = Couch(trains_capacity,
                         loads_for_trains,
                         score_condition=0,
                         epoch_condition=200,
                         steps_to_evaluate=20,
                         root_gene="LONGEST",
                         typeof="fill_bag",
                         verbose=4)
        couching.train()
        pass


    def timeplan_test() -> None:
        """
        Testing function for TimeBag, TimePlan and Couch classes

        Returns:
            None
        """
        time_bags_capacity = {"Monday": ("07:00", "21:00", 14),
                              "Tuesday": ("07:00", "21:00", 14),
                              "Wednesday": ("07:00", "21:00", 14),
                              "Thursday": ("07:00", "23:00", 16),
                              "Friday": ("07:00", "23:00", 16),
                              "Saturday": ("07:00", "23:00", 16),
                              "Sunday": ("08:00", "23:00", 15)
                              }
        to_bags_items = {"Person_1": {"Monday": ("07:00", "12:00", 5),
                                      "Tuesday": ("07:00", "12:00", 5),
                                      "Wednesday": ("07:00", "12:00", 5),
                                      },
                         "Person_2": {"Monday": ("11:00", "21:00", 4),
                                      "Tuesday": ("11:00", "21:00", 4),
                                      "Wednesday": ("11:00", "21:00", 4),
                                      },
                         "Person_3": {"Monday": ("11:00", "21:00", 5),
                                      "Tuesday": ("11:00", "21:00", 5),
                                      "Wednesday": ("11:00", "21:00", 5),
                                      },
                         "Person_4": {"Thursday": ("07:00", "12:00", 5),
                                      "Friday": ("12:00", "19:00", 5),
                                      "Saturday": ("14:00", "15:00", 1)
                                      },
                         "Person_5": {"Monday": ("15:00", "22:00", 4),
                                      "Tuesday": ("15:00", "23:00", 4),
                                      "Wednesday": ("07:00", "23:00", 4),
                                      "Thursday": ("14:00", "22:00", 5),
                                      "Friday": ("12:00", "19:00", 3),
                                      "Saturday": ("09:00", "12:00", 1),
                                      "Sunday": ("14:00", "23:00", 3),
                                      }
                         }

        couching = Couch(time_bags_capacity,
                         to_bags_items,
                         score_condition=0,
                         epoch_condition=300,
                         steps_to_evaluate=40,
                         root_gene="NORMAL",
                         typeof="time_bag",
                         verbose=4)
        couching.train()
        pass


    # fillbags test
    # shelf_test()

    # timebags test
    timeplan_test()
