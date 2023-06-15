from fal_serverless import isolated


@isolated(requirements=["pyjokes"])
def isolated_joke():
    import pyjokes

    return pyjokes.get_joke()


print(isolated_joke())