{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'numpy.core._multiarray_umath'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'numpy.core._multiarray_umath'"
     ]
    },
    {
     "ename": "SystemError",
     "evalue": "<class '_frozen_importlib._ModuleLockManager'> returned a result with an error set",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "\u001b[1;31mImportError\u001b[0m: numpy.core.multiarray failed to import",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[1;31mSystemError\u001b[0m                               Traceback (most recent call last)",
      "\u001b[1;32mE:\\anaconda\\lib\\importlib\\_bootstrap.py\u001b[0m in \u001b[0;36m_find_and_load\u001b[1;34m(name, import_)\u001b[0m\n",
      "\u001b[1;31mSystemError\u001b[0m: <class '_frozen_importlib._ModuleLockManager'> returned a result with an error set"
     ]
    },
    {
     "ename": "ImportError",
     "evalue": "numpy.core._multiarray_umath failed to import",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "\u001b[1;31mImportError\u001b[0m: numpy.core._multiarray_umath failed to import"
     ]
    },
    {
     "ename": "ImportError",
     "evalue": "numpy.core.umath failed to import",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "\u001b[1;31mImportError\u001b[0m: numpy.core.umath failed to import"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'tf' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-59639c9325fd>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m#constants\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0ma\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconstant\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0mb\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconstant\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m3\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0ma\u001b[0m \u001b[1;33m+\u001b[0m \u001b[0mb\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0msess\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mSession\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'tf' is not defined"
     ]
    }
   ],
   "source": [
    "#constants\n",
    "a = tf.constant(2)\n",
    "b = tf.constant(3)\n",
    "c = a + b\n",
    "sess = tf.Session()\n",
    "sess.run(a)\n",
    "\n",
    "a1 = tf.constant([[3,3]])\n",
    "a2 = tf.constant([[3],[3]])\n",
    "res = tf.matmul(a1, a2)\n",
    "sess.run(res)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
