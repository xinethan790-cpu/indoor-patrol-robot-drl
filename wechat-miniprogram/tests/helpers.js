'use strict';

function createManualScheduler() {
  let nextId = 1;
  const tasks = [];

  return {
    setTimeout(callback, delay) {
      const task = { id: nextId, callback, delay, cancelled: false };
      nextId += 1;
      tasks.push(task);
      return task.id;
    },

    clearTimeout(id) {
      const task = tasks.find((candidate) => candidate.id === id);
      if (task) task.cancelled = true;
    },

    runNext() {
      const index = tasks.findIndex((task) => !task.cancelled);
      if (index < 0) return null;
      const [task] = tasks.splice(index, 1);
      task.callback();
      return task;
    },

    runAll(limit) {
      const maximum = limit || 1000;
      let count = 0;
      while (this.pending().length > 0) {
        if (count >= maximum) throw new Error('scheduler task limit exceeded');
        this.runNext();
        count += 1;
      }
      return count;
    },

    pending() {
      return tasks.filter((task) => !task.cancelled);
    },
  };
}

module.exports = { createManualScheduler };
