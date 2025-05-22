from app.api import keepalive, version, faceswap

def routes_init(app):
    # 注册路由
    app.include_router(faceswap.router, tags=["faceswap"])
    app.include_router(keepalive.router, prefix="/api/v1", tags=["other"])
    app.include_router(version.router, prefix="/api/v1", tags=["other"])
    
