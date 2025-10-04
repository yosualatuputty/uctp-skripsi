
import random
import copy
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import trange
from skopt import gp_minimize
from skopt.space import Real, Integer

# Load data: mata kuliah
courses = pd.read_csv('Data/data_final.csv')
courses = courses.rename(columns={
    'Kode MK': 'code',
    'Nama MK': 'name',
    'Prodi MK': 'program',
    'Prioritas': 'priority',
    'SKS': 'duration'
})
courses['room_type'] = courses['Praktikum'].apply(lambda x: 'lab' if x == 1 else 'class')
courses['lecturer_id'] = courses['Nama Dosen'].astype('category').cat.codes
courses = courses[['code', 'name', 'program', 'priority', 'room_type', 'lecturer_id', 'duration']]
courses.index.name = 'course_id'

# Load data: Ruangan
rooms = pd.read_csv('Data/Ruang Kuliah.csv')
rooms['room_type'] = rooms['keterangan'].apply(lambda x: 'lab' if 'Riset' in x else 'class')
rooms.index = rooms['kode_ruang']

# Timeslot generation
DAYS = ['Mon','Tue','Wed','Thu','Fri']
HOURS = list(range(7,18)) # 7..17

available_slots = []
for day in DAYS:
    for h in HOURS:
        if day in ['Mon','Tue','Wed','Thu'] and h==12:
            continue
        if day=='Fri' and h in (11,12):
            continue
        available_slots.append((day,h))

SLOT_INDEX = {s:i for i,s in enumerate(available_slots)}

# Lecturer time preferences (soft constraint)
lecturer_prefs = {
    39: [(d,h) for d in DAYS for h in range(7,11)],
    103: [(d,h) for d in DAYS for h in range(13,17)],
}

# Fitness & constraint evaluation
BIG_PENALTY = 10_000

def check_hard_constraints(schedule):
    penalty = 0
    room_usage = defaultdict(list)
    for cid,(sidx,room) in schedule.items():
        duration = int(courses.loc[cid,'duration'])
        for dt in range(duration):
            idx = sidx + dt
            if idx >= len(available_slots):
                penalty += BIG_PENALTY
            else:
                room_usage[(room,idx)].append(cid)
    for k,v in room_usage.items():
        if len(v)>1:
            penalty += BIG_PENALTY * (len(v)-1)

    lecturer_usage = defaultdict(list)
    for cid,(sidx,room) in schedule.items():
        lecturer = courses.loc[cid,'lecturer_id']
        duration = int(courses.loc[cid,'duration'])
        for dt in range(duration):
            idx = sidx + dt
            lecturer_usage[(lecturer,idx)].append((cid,room))
    for k,v in lecturer_usage.items():
        if len(v)>1:
            penalty += BIG_PENALTY * (len(v)-1)

    for cid,(sidx,room) in schedule.items():
        room_type = rooms.loc[room,'room_type']
        required = courses.loc[cid,'room_type']
        if room_type != required:
            penalty += BIG_PENALTY
    return penalty

def soft_score(schedule):
    score = 0
    for cid,(sidx,room) in schedule.items():
        pr = courses.loc[cid,'priority']
        score += pr

    lec_slots = defaultdict(list)
    for cid,(sidx,room) in schedule.items():
        lec = courses.loc[cid,'lecturer_id']
        lec_slots[lec].append((sidx,room))
    for lec,assigns in lec_slots.items():
        assigns_sorted = sorted(assigns)
        for i in range(len(assigns_sorted)-1):
            s1,r1 = assigns_sorted[i]
            s2,r2 = assigns_sorted[i+1]
            if s2==s1+1:
                f1 = rooms.loc[r1,'lantai']
                f2 = rooms.loc[r2,'lantai']
                score += abs(f1-f2)

    for cid,(sidx,room) in schedule.items():
        lec = courses.loc[cid,'lecturer_id']
        slot = available_slots[sidx]
        if lec in lecturer_prefs and slot not in lecturer_prefs[lec]:
            score += 1
    return score

def fitness(schedule):
    return check_hard_constraints(schedule) + soft_score(schedule)

def random_schedule():
    sch = {}
    for cid in courses.index:
        duration = int(courses.loc[cid,'duration'])
        max_start = len(available_slots)-duration
        sidx = random.randint(0,max_start)
        req = courses.loc[cid,'room_type']
        candidate_rooms = rooms[rooms['room_type']==req].index.tolist()
        room = random.choice(candidate_rooms)
        sch[cid] = (sidx, room)
    return sch

def ga_optimize(generations=200,pop_size=50,crossover_prob=0.8,mut_prob=0.2):
    pop = [random_schedule() for _ in range(pop_size)]
    pop_scores = [fitness(ind) for ind in pop]
    best_idx = int(np.argmin(pop_scores))
    best = pop[best_idx]
    best_score = pop_scores[best_idx]
    for gen in trange(generations, desc='GA'):
        new_pop = []
        while len(new_pop)<pop_size:
            i1,i2 = random.sample(range(pop_size),2)
            parent1 = pop[i1] if pop_scores[i1]<pop_scores[i2] else pop[i2]
            i3,i4 = random.sample(range(pop_size),2)
            parent2 = pop[i3] if pop_scores[i3]<pop_scores[i4] else pop[i4]
            if random.random()<crossover_prob:
                keys = list(courses.index)
                cx = random.randint(1,len(keys)-1)
                child = {}
                for i,k in enumerate(keys):
                    child[k] = copy.deepcopy(parent1[k]) if i<cx else copy.deepcopy(parent2[k])
            else:
                child = copy.deepcopy(parent1)
            if random.random()<mut_prob:
                m = random.choice(list(courses.index))
                duration = int(courses.loc[m,'duration'])
                sidx = random.randint(0,len(available_slots)-duration)
                candidate_rooms = rooms[rooms['room_type']==courses.loc[m,'room_type']].index.tolist()
                room = random.choice(candidate_rooms)
                child[m] = (sidx,room)
            new_pop.append(child)
        pop = new_pop
        pop_scores = [fitness(ind) for ind in pop]
        cur_best_idx = int(np.argmin(pop_scores))
        if pop_scores[cur_best_idx] < best_score:
            best_score = pop_scores[cur_best_idx]
            best = pop[cur_best_idx]
    return best, best_score

# Bayesian Optimization
search_space = [
    Integer(50, 500, name='generations'),
    Integer(10, 100, name='pop_size'),
    Real(0.5, 1.0, name='crossover_prob'),
    Real(0.1, 0.5, name='mut_prob')
]

def objective(params):
    generations, pop_size, crossover_prob, mut_prob = params
    _, best_score = ga_optimize(
        generations=generations,
        pop_size=pop_size,
        crossover_prob=crossover_prob,
        mut_prob=mut_prob
    )
    return best_score

if __name__ == '__main__':
    print("Running Bayesian Optimization for parameter tuning...")
    result = gp_minimize(
        objective,
        search_space,
        n_calls=20,  # Number of evaluations
        random_state=0
    )

    best_params = result.x
    print(f"\nBest parameters found: \nGenerations: {best_params[0]}\nPopulation Size: {best_params[1]}\nCrossover Probability: {best_params[2]:.4f}\nMutation Probability: {best_params[3]:.4f}")

    print("\nRunning Genetic Algorithm with the best parameters...")
    best_schedule, best_score = ga_optimize(
        generations=best_params[0],
        pop_size=best_params[1],
        crossover_prob=best_params[2],
        mut_prob=best_params[3]
    )

    print(f"\nBest schedule found with score: {best_score}")

    rows = []
    for cid, (sidx, room) in best_schedule.items():
        timeslot = available_slots[sidx]
        rows.append((courses.loc[cid, 'code'], courses.loc[cid, 'name'], courses.loc[cid, 'lecturer_id'], timeslot[0], timeslot[1], room))
    
    df = pd.DataFrame(rows, columns=['code', 'name', 'lecturer', 'day', 'hour', 'room'])
    print(df.sort_values(['day', 'hour']).to_string(index=False))
