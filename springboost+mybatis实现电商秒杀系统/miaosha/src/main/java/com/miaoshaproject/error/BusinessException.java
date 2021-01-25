package com.miaoshaproject.error;

//程序出了任何跑不下去的异常，统一抛exception，这个exception会被最后的contoller层的springboot的
//handler捕获并做处理

//设计模式：包装器业务异常类实现
public class BusinessException extends Exception implements CommonError{

    private CommonError commonError;

    //直接接收EmBusinessError的传参用于构造业务异常
    public BusinessException(CommonError commonError){
        super(); //注意这里要调用，会有Exception自身的初始化机制在里面
        this.commonError = commonError;
    }

    //接收自定义 errMsg方式构造业务异常
    public BusinessException(CommonError commonError,String errMsg){
        super(); //注意这里要调用，会有Exception自身的初始化机制在里面
        this.commonError = commonError;
        this.commonError.setErrMsg(errMsg);
    }


    @Override
    public int getErrCode() {
        return this.commonError.getErrCode();
    }

    @Override
    public String getErrMsg() {
        return this.commonError.getErrMsg();
    }

    @Override
    public CommonError setErrMsg(String errMsg) {
        this.commonError.setErrMsg(errMsg);
        return this;
    }
}
