from multiprocessing import Pool

class Parallelization:
    def __init__(self, n_cores):
        self.n_cores = n_cores
    
    def run(self, function, data):
        if(self.n_cores < 2):
            return self.single_core_run(function, data)
        
        print("Running in parallel with ", self.n_cores, " cores")
        pool = Pool(self.n_cores)
        return pool.map(function, data)
    
    def single_core_run(self, function, data):
        print("Running in a single core")
        results = []
        for i in range(len(data)):
            result = function(data[i])
            results.append(result)
        return results