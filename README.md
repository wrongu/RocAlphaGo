# AlphaGo Replication

This project is a replication/reference implementation of DeepMind's 2016 Nature publication, "Mastering the game of Go with deep neural networks and tree search," details of which can be found [on their website](http://deepmind.com/alpha-go.html). This implementation uses Python and Keras - a decision to prioritize code clarity, at least in the early stages.

[![Build Status](https://travis-ci.org/Rochester-NRT/AlphaGo.svg?branch=develop)](https://travis-ci.org/Rochester-NRT/AlphaGo)
[![Gitter](https://badges.gitter.im/Rochester-NRT/AlphaGo.svg)](https://gitter.im/Rochester-NRT/AlphaGo?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# Current project status

_This is not yet a full implementation of AlphaGo_. Development is being carried out on the `develop` branch.

We are still early in development. There are quite a few pieces to AlphaGo that can be written in parallel. We are currently focusing our efforts on the supervised and "self-play" parts of the training pipeline because the training itself may take a very long time..

> Updates were applied asynchronously on 50 GPUs... Training took around 3 weeks for 340 million training steps
>
> -Silver et al. (page 8)

See the wiki page on the [training pipeline](https://github.com/Rochester-NRT/AlphaGo/wiki/Neural-Networks-and-Training) for information on how to run the training commands.

# How to contribute

See the ['Contributing'](CONTRIBUTING.md) document and join the [Gitter chat](https://gitter.im/Rochester-NRT/AlphaGo?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge).
