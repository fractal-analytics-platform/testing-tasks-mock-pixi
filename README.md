# testing-tasks-mock-pixi

How to create a tag and trigger the GitHub-release action:
```
git tag -a 0.0.1
git push --tags 
```

How to update manifest
```
pixi run fractal-manifest create --package mock-pytorch-pixi-tasks
```
