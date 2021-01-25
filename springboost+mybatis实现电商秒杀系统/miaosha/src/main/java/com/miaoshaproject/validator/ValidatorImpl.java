package com.miaoshaproject.validator;

import org.springframework.beans.factory.InitializingBean;
import org.springframework.stereotype.Component;

import javax.validation.ConstraintViolation;
import javax.validation.Validation;
import javax.validation.Validator; //注意不要引入错了 import javax.xml.validation.Validator;
import java.util.Set;

@Component //表明一个spring的bean，在类扫描的时候会扫描到
public class ValidatorImpl implements InitializingBean {

    //要包装出来Validtor类的validtor的工具
    private Validator validator;

    //实现校验方法并返回校验结果
    public ValidationResult validate(Object bean){//可以校验任何bean对象，返回上面定义的ValidationResult
        final ValidationResult result = new ValidationResult();
        //由于ValidationResult的属性已经被初始化，不用担心有空指针错误
        Set<ConstraintViolation<Object>> constraintViolationSet = validator.validate(bean);

        if (constraintViolationSet.size() > 0) {//大于0 有错误
            result.setHasErrors(true);

            //需要设置jdk8 的 lambda表达式 否则会编译错误
            constraintViolationSet.forEach(constraintViolation -> {
                String errMsg = constraintViolation.getMessage();
                String propertyName = constraintViolation.getPropertyPath().toString();
                result.getErrorMsgMap().put(propertyName,errMsg);
            });
        }

        return result;
    }

    @Override
    public void afterPropertiesSet() throws Exception {
        //在springBean初始化完成之后会回调ValidatorImpl 的afterPropertiesSet

        //将hibernate validator通过工厂的初始化方式使其实例化
        this.validator = (Validator) Validation.buildDefaultValidatorFactory().getValidator();
    }
}
