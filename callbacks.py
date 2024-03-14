import time
from pytorch_lightning.callbacks import Callback


class PrintingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.start = time.time()
        print('Epoch: %4d, ' % trainer.current_epoch, flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if len(pl_module.history['loss']) == 0:
            return
        loss = pl_module.history['loss']
        avg_loss = sum(loss) / len(loss)
        pl_module.history['loss'] = []
        print('Total time: %4ds, loss: %2.4f' % (int(time.time() - self.start), avg_loss), flush=True)
        print('-' * 50, flush=True)

    def on_validation_start(self, trainer, pl_module):
        if hasattr(self, 'start'):
            print('Training time: %4ds' % (int(time.time() - self.start)), flush=True)
        self.val_start = time.time()

    def on_validation_end(self, trainer, pl_module):
        print(pl_module.history['perf'], flush=True)
        print('Validation time: %4ds' % (int(time.time() - self.val_start)), flush=True)

    def on_test_end(self, trainer, pl_module):
        print(flush=True)
        print('=' * 50, flush=True)
        print('Epoch: test', flush=True)
        print(pl_module.history['perf'], flush=True)
        print('=' * 50, flush=True)

class TimeCallback(Callback):  
    def on_train_epoch_start(self, trainer, pl_module):  
        self.epoch_start_time = time.time()  
  
    def on_train_epoch_end(self, trainer, pl_module, outputs):  
        epoch_end_time = time.time()  
        epoch_time = epoch_end_time - self.epoch_start_time  
        print(f"Epoch took: {epoch_time:.2f} seconds")  
  
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):  
        self.batch_start_time = time.time()  
  
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  
        batch_end_time = time.time()  
        batch_time = batch_end_time - self.batch_start_time  
        print(f"Batch {batch_idx} took: {batch_time:.2f} seconds")  
  
    def on_after_backward(self, trainer, pl_module):  
        backward_end_time = time.time()  
        backward_time = backward_end_time - self.backward_start_time  
        print(f"Backward took: {backward_time:.2f} seconds")
  
    def on_before_zero_grad(self, trainer, pl_module, optimizer):    
        self.backward_start_time = time.time()  
  
    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx):  
        self.optimizer_start_time = time.time()  
  
    def on_after_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx):  
        optimizer_end_time = time.time()  
        optimizer_time = optimizer_end_time - self.optimizer_start_time  
        print(f"Optimizer step took: {optimizer_time:.2f} seconds")  
  
    def on_validation_epoch_start(self, trainer, pl_module):  
        self.val_epoch_start_time = time.time()  
  
    def on_validation_epoch_end(self, trainer, pl_module):  
        val_epoch_end_time = time.time()  
        val_epoch_time = val_epoch_end_time - self.val_epoch_start_time  
        print(f"Validation epoch took: {val_epoch_time:.2f} seconds")  
  
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):  
        self.val_batch_start_time = time.time()  
  
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  
        val_batch_end_time = time.time()  
        val_batch_time = val_batch_end_time - self.val_batch_start_time  
        print(f"Validation batch {batch_idx} took: {val_batch_time:.2f} seconds")  