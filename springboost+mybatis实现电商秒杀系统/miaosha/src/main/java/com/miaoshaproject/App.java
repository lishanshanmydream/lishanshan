package com.miaoshaproject;

import com.miaoshaproject.dao.UserDOMapper;
import com.miaoshaproject.dataobject.UserDO;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * https://spring.io/projects/spring-boot
 * 1.https://spring.io/guides/gs/rest-service/ 示例创建一个基于restful的web服务
 * 先找到pom  指定父pom： org.springframework.boot ，然后我们使用了父pom spring-boot-starter-parent 中的一个spring-boot-starter-web的项目
 * （会下载这个项目gav坐标对应的jar包）以此使用springboot开源的项目依赖关系，构建自己的应用
 *
 * 2.使用springboot搭建web项目
 * 1）使用EnableAutoConfiguration springboot可以自动加载一个内嵌的tomcat并加载默认配置
 * 2）声明RestController + RequestMapping可以实现springmvc之前要配置severlet配置web.xml等复杂的功能
 * 备注：在resources里可以创建application.properties 改变springboot默认配置
 *
 * 3.Mybatis接入SpringBoot:将依赖包都引入
 * 1）数据库使用mysql 因此将mysql依赖的client的jdbc配置加载进来mysql-connector-java
 * 2）连接池管理mysql连接 采用druid链接池
 * 3）springboot 对mybatis的支持下载: mybatis-spring-boot-starter
 * 4）在application.properties中导入mybatis的配置 用mybatis-generator生成 dataobject和dao
 *
 * 4. springmvc
 * service-model层对外暴露可调用的用接口，增加impl为service接口的实现，model为业务模型（dao是数据模型层面）（增加contoller 和 dao直接的逻辑处理，不能直接到dao的实例返回）
 *  controller层（增加viewobject）不能直接将model的业务模型直接个前端需要转为viewobject 将可以给前端的字段给前端
 *
 * 5. 异常处理
 *
 */

//@EnableAutoConfiguration //将App的启动类当成一个自动化可以自动支持配置的Bean。并开启基于springboot的自动化的配置//springboot项目会将我们所有对数据库的依赖、对redis的依赖或springtom容器本身的一些（aop）依赖的管理 统统以自动化的配置加载到工程当中。
@SpringBootApplication(scanBasePackages = {"com.miaoshaproject"}) //秒杀对应的根目录下的包自动做扫描
@RestController//springmvc用来解决web控制层的一些问题。通过springboot配置搞定mvc contoller功能
@MapperScan("com.miaoshaproject.dao")  //dao存放的地方放置在这个注解下
public class App
{
    @Autowired
    private UserDOMapper userDOMapper;

    @RequestMapping("/") //配合restcontrol 当用于访问根目录的时候
    public  String home() {
        UserDO userDO = userDOMapper.selectByPrimaryKey(1);
        if(userDO == null) {
            return "用户对象不存在!";
        }else{
            return userDO.getName();
        }

    }

    public static void main( String[] args )
    {
        System.out.println( "Hello World!" );
        SpringApplication.run(App.class,args); //一旦开启了EnableAutoConfiguration，需要用这行代码启动springboost的项目
        //默认启动了一个内嵌的web的tocat容器 并在8080端口被监听 http://localhost:8080
    }
}
