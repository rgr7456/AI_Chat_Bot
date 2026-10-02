from app.database.database import AsyncSessionLocal

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# async def get_db():
#     async with AsyncSessionLocal() as session:
#         try:
#             yield session
#             await session.commit()   # Commit only if everything works
#         except:
#             await session.rollback() # Rollback on any error
#             raise
#         finally:
#             await session.close()